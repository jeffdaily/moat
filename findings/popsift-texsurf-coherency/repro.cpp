// Minimal standalone HIP reproducer for the popsift texture/surface coherency
// problem seen on gfx90a (MI250X, CDNA2).
//
// Mirrors popsift's exact pattern:
//   - layered float cudaArray via hipMalloc3DArray(Layered | SurfaceLoadStore)
//   - a texture object (point filter, element-type read) created ONCE before any
//     surface write, then reused
//   - kernel A: surf2DLayeredwrite a known value
//   - kernel B (SEPARATE launch, SAME stream): tex2DLayered read it back
//
// Per the CUDA C Programming Guide, the texture cache is invalidated at every
// kernel-launch boundary, so kernel B (a later launch) must observe kernel A's
// writes. This program checks whether HIP on gfx90a honors that.
//
// Build: hipcc --offload-arch=gfx90a -O2 repro.cpp -o repro

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>

#define CHECK(cmd)                                                             \
    do {                                                                       \
        hipError_t e = (cmd);                                                  \
        if (e != hipSuccess) {                                                 \
            printf("HIP error %s:%d: %s\n", __FILE__, __LINE__,               \
                   hipGetErrorString(e));                                      \
            return -1;                                                         \
        }                                                                      \
    } while (0)

static const int W = 64;
static const int H = 64;
static const int L = 4;            // layers
static const float MAGIC = 161.7f; // the popsift value
static const int TEST_LAYER = 2;

// ---- Layered kernels (popsift uses tex2DLayered / surf2DLayeredwrite) ----

__global__ void writeSurfLayered(hipSurfaceObject_t surf, float v) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    // popsift writes byte offset x*4 (sizeof(float)) on the x axis
    surf2DLayeredwrite(v, surf, x * 4, y, TEST_LAYER);
}

__global__ void readTexLayered(hipTextureObject_t tex, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    // popsift readTex: tex2DLayered<float>(tex, x+0.5, y+0.5, layer) (point filter)
    out[y * W + x] = tex2DLayered<float>(tex, x + 0.5f, y + 0.5f, TEST_LAYER);
}

__global__ void readSurfLayered(hipSurfaceObject_t surf, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float v;
    surf2DLayeredread(&v, surf, x * 4, y, TEST_LAYER);
    out[y * W + x] = v;
}

// ---- Non-layered 2D kernels (contrast) ----

__global__ void writeSurf2D(hipSurfaceObject_t surf, float v) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    surf2Dwrite(v, surf, x * 4, y);
}

__global__ void readTex2D(hipTextureObject_t tex, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    out[y * W + x] = tex2D<float>(tex, x + 0.5f, y + 0.5f);
}

struct Stats {
    float minv, maxv, sample;
    int fresh; // count of cells == MAGIC
    int zero;  // count of cells == 0
};

static Stats summarize(const float* h, int n) {
    Stats s{1e30f, -1e30f, h[TEST_LAYER /*dummy*/], 0, 0};
    s.sample = h[(H / 2) * W + (W / 2)];
    for (int i = 0; i < n; i++) {
        if (h[i] < s.minv) s.minv = h[i];
        if (h[i] > s.maxv) s.maxv = h[i];
        if (h[i] == MAGIC) s.fresh++;
        if (h[i] == 0.0f) s.zero++;
    }
    return s;
}

static void report(const char* name, const Stats& s, int n) {
    const char* verdict;
    if (s.fresh == n)
        verdict = "FRESH (all == magic)";
    else if (s.zero == n)
        verdict = "STALE (all == 0)";
    else
        verdict = "MIXED";
    printf("  %-46s min=%.4f max=%.4f center=%.4f  -> %s\n", name, s.minv,
           s.maxv, s.sample, verdict);
}

// Helpers to (re)create a layered array's tex/surf.
struct LayeredRes {
    hipArray_t arr = nullptr;
    hipSurfaceObject_t surf = 0;
    hipTextureObject_t tex = 0;
};

static int makeLayeredArray(hipArray_t* arr) {
    hipChannelFormatDesc desc = hipCreateChannelDesc(32, 0, 0, 0,
                                                     hipChannelFormatKindFloat);
    hipExtent ext = make_hipExtent(W, H, L); // width in elements for arrays
    CHECK(hipMalloc3DArray(arr, &desc, ext,
                           hipArrayLayered | hipArraySurfaceLoadStore));
    return 0;
}

static int makeSurf(hipArray_t arr, hipSurfaceObject_t* surf) {
    hipResourceDesc rd;
    memset(&rd, 0, sizeof(rd));
    rd.resType = hipResourceTypeArray;
    rd.res.array.array = arr;
    CHECK(hipCreateSurfaceObject(surf, &rd));
    return 0;
}

static int makeTexPoint(hipArray_t arr, hipTextureObject_t* tex) {
    hipResourceDesc rd;
    memset(&rd, 0, sizeof(rd));
    rd.resType = hipResourceTypeArray;
    rd.res.array.array = arr;
    hipTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = hipAddressModeClamp;
    td.addressMode[1] = hipAddressModeClamp;
    td.addressMode[2] = hipAddressModeClamp;
    td.readMode = hipReadModeElementType;
    td.filterMode = hipFilterModePoint;
    CHECK(hipCreateTextureObject(tex, &rd, &td, nullptr));
    return 0;
}

int main() {
    int dev = 0;
    CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, dev));
    printf("Device %d: %s (gcnArch %s)\n", dev, prop.name, prop.gcnArchName);
    printf("W=%d H=%d L=%d magic=%.4f test_layer=%d\n\n", W, H, L, MAGIC,
           TEST_LAYER);

    const int N = W * H;
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    float* d_out = nullptr;
    CHECK(hipMalloc(&d_out, N * sizeof(float)));
    float* h_out = (float*)malloc(N * sizeof(float));

    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));

    // =====================================================================
    // Variation 0 (popsift-faithful): layered array, tex created ONCE before
    // the write, separate launches on the same stream, NO sync between them,
    // read via TEXTURE.
    // =====================================================================
    printf("[A] Layered: tex created BEFORE write, read via TEXTURE, no sync between launches\n");
    {
        LayeredRes r;
        if (makeLayeredArray(&r.arr)) return -1;
        if (makeSurf(r.arr, &r.surf)) return -1;
        if (makeTexPoint(r.arr, &r.tex)) return -1; // created before any write

        // zero the array first via surface (so 'stale' would read 0 like popsift)
        writeSurfLayered<<<grid, block, 0, stream>>>(r.surf, 0.0f);
        // ensure that the zeroing is itself flushed/visible before the real test
        CHECK(hipStreamSynchronize(stream));

        writeSurfLayered<<<grid, block, 0, stream>>>(r.surf, MAGIC);
        readTexLayered<<<grid, block, 0, stream>>>(r.tex, d_out);
        CHECK(hipMemcpyAsync(h_out, d_out, N * sizeof(float),
                             hipMemcpyDeviceToHost, stream));
        CHECK(hipStreamSynchronize(stream));
        report("read via tex2DLayered (reused tex)", summarize(h_out, N), N);

        CHECK(hipDestroyTextureObject(r.tex));
        CHECK(hipDestroySurfaceObject(r.surf));
        CHECK(hipFreeArray(r.arr));
    }

    // =====================================================================
    // Variation (a): same as A but read via SURFACE in kernel B (popsift's
    // surf2DLayeredread sanity check returns fresh).
    // =====================================================================
    printf("\n[B] Layered: read via SURFACE (surf2DLayeredread), no sync between launches\n");
    {
        LayeredRes r;
        if (makeLayeredArray(&r.arr)) return -1;
        if (makeSurf(r.arr, &r.surf)) return -1;
        if (makeTexPoint(r.arr, &r.tex)) return -1;

        writeSurfLayered<<<grid, block, 0, stream>>>(r.surf, 0.0f);
        CHECK(hipStreamSynchronize(stream));

        writeSurfLayered<<<grid, block, 0, stream>>>(r.surf, MAGIC);
        readSurfLayered<<<grid, block, 0, stream>>>(r.surf, d_out);
        CHECK(hipMemcpyAsync(h_out, d_out, N * sizeof(float),
                             hipMemcpyDeviceToHost, stream));
        CHECK(hipStreamSynchronize(stream));
        report("read via surf2DLayeredread", summarize(h_out, N), N);

        CHECK(hipDestroyTextureObject(r.tex));
        CHECK(hipDestroySurfaceObject(r.surf));
        CHECK(hipFreeArray(r.arr));
    }

    // =====================================================================
    // Variation (b): recreate the texture object AFTER the write, before B.
    // =====================================================================
    printf("\n[C] Layered: tex created AFTER write (fresh tex), read via TEXTURE\n");
    {
        LayeredRes r;
        if (makeLayeredArray(&r.arr)) return -1;
        if (makeSurf(r.arr, &r.surf)) return -1;

        writeSurfLayered<<<grid, block, 0, stream>>>(r.surf, 0.0f);
        CHECK(hipStreamSynchronize(stream));

        writeSurfLayered<<<grid, block, 0, stream>>>(r.surf, MAGIC);
        // create the texture only now, after the write
        if (makeTexPoint(r.arr, &r.tex)) return -1;
        readTexLayered<<<grid, block, 0, stream>>>(r.tex, d_out);
        CHECK(hipMemcpyAsync(h_out, d_out, N * sizeof(float),
                             hipMemcpyDeviceToHost, stream));
        CHECK(hipStreamSynchronize(stream));
        report("read via tex2DLayered (fresh tex)", summarize(h_out, N), N);

        CHECK(hipDestroyTextureObject(r.tex));
        CHECK(hipDestroySurfaceObject(r.surf));
        CHECK(hipFreeArray(r.arr));
    }

    // =====================================================================
    // Variation (d): reused tex, but explicit hipDeviceSynchronize between A and B.
    // =====================================================================
    printf("\n[D] Layered: reused tex, explicit hipDeviceSynchronize between launches\n");
    {
        LayeredRes r;
        if (makeLayeredArray(&r.arr)) return -1;
        if (makeSurf(r.arr, &r.surf)) return -1;
        if (makeTexPoint(r.arr, &r.tex)) return -1;

        writeSurfLayered<<<grid, block, 0, stream>>>(r.surf, 0.0f);
        CHECK(hipStreamSynchronize(stream));

        writeSurfLayered<<<grid, block, 0, stream>>>(r.surf, MAGIC);
        CHECK(hipStreamSynchronize(stream)); // explicit sync (host) between launches
        CHECK(hipDeviceSynchronize());
        readTexLayered<<<grid, block, 0, stream>>>(r.tex, d_out);
        CHECK(hipMemcpyAsync(h_out, d_out, N * sizeof(float),
                             hipMemcpyDeviceToHost, stream));
        CHECK(hipStreamSynchronize(stream));
        report("read via tex2DLayered (reused tex, synced)", summarize(h_out, N), N);

        CHECK(hipDestroyTextureObject(r.tex));
        CHECK(hipDestroySurfaceObject(r.surf));
        CHECK(hipFreeArray(r.arr));
    }

    // =====================================================================
    // Variation (e): reused tex across a SECOND write+read round-trip. This
    // tests whether the FIRST kernel-launch boundary primed the cache and a
    // later write is then invisible (the popsift multi-frame reuse case).
    // =====================================================================
    printf("\n[E] Layered: reused tex, two rounds (write v1->read, write v2->read)\n");
    {
        LayeredRes r;
        if (makeLayeredArray(&r.arr)) return -1;
        if (makeSurf(r.arr, &r.surf)) return -1;
        if (makeTexPoint(r.arr, &r.tex)) return -1;

        // round 1: write MAGIC, read via tex (prime the cache)
        writeSurfLayered<<<grid, block, 0, stream>>>(r.surf, MAGIC);
        readTexLayered<<<grid, block, 0, stream>>>(r.tex, d_out);
        CHECK(hipStreamSynchronize(stream));
        CHECK(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));
        report("round1 (write magic, read tex)", summarize(h_out, N), N);

        // round 2: overwrite with a DIFFERENT value, read via the SAME tex
        const float MAGIC2 = 42.0f;
        writeSurfLayered<<<grid, block, 0, stream>>>(r.surf, MAGIC2);
        readTexLayered<<<grid, block, 0, stream>>>(r.tex, d_out);
        CHECK(hipStreamSynchronize(stream));
        CHECK(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));
        Stats s = summarize(h_out, N);
        int isM2 = 0;
        for (int i = 0; i < N; i++) if (h_out[i] == MAGIC2) isM2++;
        printf("  %-46s min=%.4f max=%.4f center=%.4f  -> %s\n",
               "round2 (write 42, read SAME tex)", s.minv, s.maxv, s.sample,
               isM2 == N ? "FRESH (==42)"
               : s.fresh == N ? "STALE (still ==161.7)"
                              : "MIXED");

        CHECK(hipDestroyTextureObject(r.tex));
        CHECK(hipDestroySurfaceObject(r.surf));
        CHECK(hipFreeArray(r.arr));
    }

    // =====================================================================
    // Variation (c): NON-layered 2D array contrast (surf2Dwrite/tex2D).
    // =====================================================================
    printf("\n[F] Non-layered 2D: tex created BEFORE write, read via TEXTURE\n");
    {
        hipChannelFormatDesc desc =
            hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
        hipArray_t arr;
        CHECK(hipMallocArray(&arr, &desc, W, H, hipArraySurfaceLoadStore));
        hipResourceDesc rd;
        memset(&rd, 0, sizeof(rd));
        rd.resType = hipResourceTypeArray;
        rd.res.array.array = arr;
        hipSurfaceObject_t surf;
        CHECK(hipCreateSurfaceObject(&surf, &rd));
        hipTextureDesc td;
        memset(&td, 0, sizeof(td));
        td.normalizedCoords = 0;
        td.addressMode[0] = hipAddressModeClamp;
        td.addressMode[1] = hipAddressModeClamp;
        td.readMode = hipReadModeElementType;
        td.filterMode = hipFilterModePoint;
        hipTextureObject_t tex;
        CHECK(hipCreateTextureObject(&tex, &rd, &td, nullptr));

        writeSurf2D<<<grid, block, 0, stream>>>(surf, 0.0f);
        CHECK(hipStreamSynchronize(stream));
        writeSurf2D<<<grid, block, 0, stream>>>(surf, MAGIC);
        readTex2D<<<grid, block, 0, stream>>>(tex, d_out);
        CHECK(hipMemcpyAsync(h_out, d_out, N * sizeof(float),
                             hipMemcpyDeviceToHost, stream));
        CHECK(hipStreamSynchronize(stream));
        report("read via tex2D (reused tex, 2D)", summarize(h_out, N), N);

        CHECK(hipDestroyTextureObject(tex));
        CHECK(hipDestroySurfaceObject(surf));
        CHECK(hipFreeArray(arr));
    }

    CHECK(hipStreamDestroy(stream));
    CHECK(hipFree(d_out));
    free(h_out);
    printf("\nDone.\n");
    return 0;
}
