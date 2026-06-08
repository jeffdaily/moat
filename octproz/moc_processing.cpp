/****************************************************************************
** Meta object code from reading C++ file 'processing.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../projects/OCTproZ/src/octproz_project/octproz/src/processing.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'processing.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_Processing_t {
    QByteArrayData data[63];
    char stringdata0[1148];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Processing_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Processing_t qt_meta_stringdata_Processing = {
    {
QT_MOC_LITERAL(0, 0, 10), // "Processing"
QT_MOC_LITERAL(1, 11, 18), // "initializationDone"
QT_MOC_LITERAL(2, 30, 0), // ""
QT_MOC_LITERAL(3, 31, 20), // "initializationFailed"
QT_MOC_LITERAL(4, 52, 10), // "initOpenGL"
QT_MOC_LITERAL(5, 63, 15), // "QOpenGLContext*"
QT_MOC_LITERAL(6, 79, 17), // "processingContext"
QT_MOC_LITERAL(7, 97, 18), // "QOffscreenSurface*"
QT_MOC_LITERAL(8, 116, 17), // "processingSurface"
QT_MOC_LITERAL(9, 134, 8), // "QThread*"
QT_MOC_LITERAL(10, 143, 16), // "processingThread"
QT_MOC_LITERAL(11, 160, 20), // "initOpenGLenFaceView"
QT_MOC_LITERAL(12, 181, 15), // "initRawRecorder"
QT_MOC_LITERAL(13, 197, 39), // "OctAlgorithmParameters::Recor..."
QT_MOC_LITERAL(14, 237, 6), // "params"
QT_MOC_LITERAL(15, 244, 21), // "initProcessedRecorder"
QT_MOC_LITERAL(16, 266, 14), // "processingDone"
QT_MOC_LITERAL(17, 281, 22), // "streamingBufferEnabled"
QT_MOC_LITERAL(18, 304, 7), // "enabled"
QT_MOC_LITERAL(19, 312, 19), // "processedRecordDone"
QT_MOC_LITERAL(20, 332, 13), // "rawRecordDone"
QT_MOC_LITERAL(21, 346, 7), // "rawData"
QT_MOC_LITERAL(22, 354, 9), // "rawBuffer"
QT_MOC_LITERAL(23, 364, 8), // "bitDepth"
QT_MOC_LITERAL(24, 373, 14), // "samplesPerLine"
QT_MOC_LITERAL(25, 388, 13), // "linesPerFrame"
QT_MOC_LITERAL(26, 402, 15), // "framesPerBuffer"
QT_MOC_LITERAL(27, 418, 16), // "buffersPerVolume"
QT_MOC_LITERAL(28, 435, 15), // "currentBufferNr"
QT_MOC_LITERAL(29, 451, 4), // "info"
QT_MOC_LITERAL(30, 456, 5), // "error"
QT_MOC_LITERAL(31, 462, 13), // "updateInfoBox"
QT_MOC_LITERAL(32, 476, 16), // "volumesPerSecond"
QT_MOC_LITERAL(33, 493, 16), // "buffersPerSecond"
QT_MOC_LITERAL(34, 510, 15), // "bscansPerSecond"
QT_MOC_LITERAL(35, 526, 15), // "ascansPerSecond"
QT_MOC_LITERAL(36, 542, 12), // "bufferSizeMB"
QT_MOC_LITERAL(37, 555, 14), // "dataThroughput"
QT_MOC_LITERAL(38, 570, 10), // "slot_start"
QT_MOC_LITERAL(39, 581, 18), // "AcquisitionSystem*"
QT_MOC_LITERAL(40, 600, 6), // "system"
QT_MOC_LITERAL(41, 607, 20), // "slot_enableRecording"
QT_MOC_LITERAL(42, 628, 9), // "recParams"
QT_MOC_LITERAL(43, 638, 30), // "slot_updateDisplayedBscanFrame"
QT_MOC_LITERAL(44, 669, 7), // "frameNr"
QT_MOC_LITERAL(45, 677, 21), // "displayFunctionFrames"
QT_MOC_LITERAL(46, 699, 15), // "displayFunction"
QT_MOC_LITERAL(47, 715, 31), // "slot_updateDisplayedEnFaceFrame"
QT_MOC_LITERAL(48, 747, 38), // "slot_registerBscanOpenGLbuffe..."
QT_MOC_LITERAL(49, 786, 14), // "openGLbufferId"
QT_MOC_LITERAL(50, 801, 43), // "slot_registerEnFaceViewOpenGL..."
QT_MOC_LITERAL(51, 845, 43), // "slot_registerVolumeViewOpenGL..."
QT_MOC_LITERAL(52, 889, 23), // "enableGpu2HostStreaming"
QT_MOC_LITERAL(53, 913, 15), // "enableStreaming"
QT_MOC_LITERAL(54, 929, 28), // "enableFloatGpu2HostStreaming"
QT_MOC_LITERAL(55, 958, 28), // "registerStreamingHostBuffers"
QT_MOC_LITERAL(56, 987, 18), // "h_streamingBuffer1"
QT_MOC_LITERAL(57, 1006, 18), // "h_streamingBuffer2"
QT_MOC_LITERAL(58, 1025, 6), // "size_t"
QT_MOC_LITERAL(59, 1032, 14), // "bytesPerBuffer"
QT_MOC_LITERAL(60, 1047, 30), // "unregisterStreamingHostBuffers"
QT_MOC_LITERAL(61, 1078, 33), // "registerFloatStreamingHostBuf..."
QT_MOC_LITERAL(62, 1112, 35) // "unregisterFloatStreamingHostB..."

    },
    "Processing\0initializationDone\0\0"
    "initializationFailed\0initOpenGL\0"
    "QOpenGLContext*\0processingContext\0"
    "QOffscreenSurface*\0processingSurface\0"
    "QThread*\0processingThread\0"
    "initOpenGLenFaceView\0initRawRecorder\0"
    "OctAlgorithmParameters::RecordingParams\0"
    "params\0initProcessedRecorder\0"
    "processingDone\0streamingBufferEnabled\0"
    "enabled\0processedRecordDone\0rawRecordDone\0"
    "rawData\0rawBuffer\0bitDepth\0samplesPerLine\0"
    "linesPerFrame\0framesPerBuffer\0"
    "buffersPerVolume\0currentBufferNr\0info\0"
    "error\0updateInfoBox\0volumesPerSecond\0"
    "buffersPerSecond\0bscansPerSecond\0"
    "ascansPerSecond\0bufferSizeMB\0"
    "dataThroughput\0slot_start\0AcquisitionSystem*\0"
    "system\0slot_enableRecording\0recParams\0"
    "slot_updateDisplayedBscanFrame\0frameNr\0"
    "displayFunctionFrames\0displayFunction\0"
    "slot_updateDisplayedEnFaceFrame\0"
    "slot_registerBscanOpenGLbufferWithCuda\0"
    "openGLbufferId\0"
    "slot_registerEnFaceViewOpenGLbufferWithCuda\0"
    "slot_registerVolumeViewOpenGLbufferWithCuda\0"
    "enableGpu2HostStreaming\0enableStreaming\0"
    "enableFloatGpu2HostStreaming\0"
    "registerStreamingHostBuffers\0"
    "h_streamingBuffer1\0h_streamingBuffer2\0"
    "size_t\0bytesPerBuffer\0"
    "unregisterStreamingHostBuffers\0"
    "registerFloatStreamingHostBuffers\0"
    "unregisterFloatStreamingHostBuffers"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Processing[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      27,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
      14,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,  149,    2, 0x06 /* Public */,
       3,    0,  150,    2, 0x06 /* Public */,
       4,    3,  151,    2, 0x06 /* Public */,
      11,    0,  158,    2, 0x06 /* Public */,
      12,    1,  159,    2, 0x06 /* Public */,
      15,    1,  162,    2, 0x06 /* Public */,
      16,    0,  165,    2, 0x06 /* Public */,
      17,    1,  166,    2, 0x06 /* Public */,
      19,    0,  169,    2, 0x06 /* Public */,
      20,    0,  170,    2, 0x06 /* Public */,
      21,    7,  171,    2, 0x06 /* Public */,
      29,    1,  186,    2, 0x06 /* Public */,
      30,    1,  189,    2, 0x06 /* Public */,
      31,    6,  192,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      38,    1,  205,    2, 0x0a /* Public */,
      41,    1,  208,    2, 0x0a /* Public */,
      43,    3,  211,    2, 0x0a /* Public */,
      47,    3,  218,    2, 0x0a /* Public */,
      48,    1,  225,    2, 0x0a /* Public */,
      50,    1,  228,    2, 0x0a /* Public */,
      51,    1,  231,    2, 0x0a /* Public */,
      52,    1,  234,    2, 0x0a /* Public */,
      54,    1,  237,    2, 0x0a /* Public */,
      55,    3,  240,    2, 0x0a /* Public */,
      60,    0,  247,    2, 0x0a /* Public */,
      61,    3,  248,    2, 0x0a /* Public */,
      62,    0,  255,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 5, 0x80000000 | 7, 0x80000000 | 9,    6,    8,   10,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 13,   14,
    QMetaType::Void, 0x80000000 | 13,   14,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   18,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::VoidStar, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt,   22,   23,   24,   25,   26,   27,   28,
    QMetaType::Void, QMetaType::QString,   29,
    QMetaType::Void, QMetaType::QString,   30,
    QMetaType::Void, QMetaType::QString, QMetaType::QString, QMetaType::QString, QMetaType::QString, QMetaType::QString, QMetaType::QString,   32,   33,   34,   35,   36,   37,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 39,   40,
    QMetaType::Void, 0x80000000 | 13,   42,
    QMetaType::Void, QMetaType::UInt, QMetaType::UInt, QMetaType::Int,   44,   45,   46,
    QMetaType::Void, QMetaType::UInt, QMetaType::UInt, QMetaType::Int,   44,   45,   46,
    QMetaType::Void, QMetaType::UInt,   49,
    QMetaType::Void, QMetaType::UInt,   49,
    QMetaType::Void, QMetaType::UInt,   49,
    QMetaType::Void, QMetaType::Bool,   53,
    QMetaType::Void, QMetaType::Bool,   53,
    QMetaType::Void, QMetaType::VoidStar, QMetaType::VoidStar, 0x80000000 | 58,   56,   57,   59,
    QMetaType::Void,
    QMetaType::Void, QMetaType::VoidStar, QMetaType::VoidStar, 0x80000000 | 58,   56,   57,   59,
    QMetaType::Void,

       0        // eod
};

void Processing::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<Processing *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->initializationDone(); break;
        case 1: _t->initializationFailed(); break;
        case 2: _t->initOpenGL((*reinterpret_cast< QOpenGLContext*(*)>(_a[1])),(*reinterpret_cast< QOffscreenSurface*(*)>(_a[2])),(*reinterpret_cast< QThread*(*)>(_a[3]))); break;
        case 3: _t->initOpenGLenFaceView(); break;
        case 4: _t->initRawRecorder((*reinterpret_cast< OctAlgorithmParameters::RecordingParams(*)>(_a[1]))); break;
        case 5: _t->initProcessedRecorder((*reinterpret_cast< OctAlgorithmParameters::RecordingParams(*)>(_a[1]))); break;
        case 6: _t->processingDone(); break;
        case 7: _t->streamingBufferEnabled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 8: _t->processedRecordDone(); break;
        case 9: _t->rawRecordDone(); break;
        case 10: _t->rawData((*reinterpret_cast< void*(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< uint(*)>(_a[3])),(*reinterpret_cast< uint(*)>(_a[4])),(*reinterpret_cast< uint(*)>(_a[5])),(*reinterpret_cast< uint(*)>(_a[6])),(*reinterpret_cast< uint(*)>(_a[7]))); break;
        case 11: _t->info((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 12: _t->error((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 13: _t->updateInfoBox((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3])),(*reinterpret_cast< QString(*)>(_a[4])),(*reinterpret_cast< QString(*)>(_a[5])),(*reinterpret_cast< QString(*)>(_a[6]))); break;
        case 14: _t->slot_start((*reinterpret_cast< AcquisitionSystem*(*)>(_a[1]))); break;
        case 15: _t->slot_enableRecording((*reinterpret_cast< OctAlgorithmParameters::RecordingParams(*)>(_a[1]))); break;
        case 16: _t->slot_updateDisplayedBscanFrame((*reinterpret_cast< uint(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 17: _t->slot_updateDisplayedEnFaceFrame((*reinterpret_cast< uint(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 18: _t->slot_registerBscanOpenGLbufferWithCuda((*reinterpret_cast< uint(*)>(_a[1]))); break;
        case 19: _t->slot_registerEnFaceViewOpenGLbufferWithCuda((*reinterpret_cast< uint(*)>(_a[1]))); break;
        case 20: _t->slot_registerVolumeViewOpenGLbufferWithCuda((*reinterpret_cast< uint(*)>(_a[1]))); break;
        case 21: _t->enableGpu2HostStreaming((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 22: _t->enableFloatGpu2HostStreaming((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 23: _t->registerStreamingHostBuffers((*reinterpret_cast< void*(*)>(_a[1])),(*reinterpret_cast< void*(*)>(_a[2])),(*reinterpret_cast< size_t(*)>(_a[3]))); break;
        case 24: _t->unregisterStreamingHostBuffers(); break;
        case 25: _t->registerFloatStreamingHostBuffers((*reinterpret_cast< void*(*)>(_a[1])),(*reinterpret_cast< void*(*)>(_a[2])),(*reinterpret_cast< size_t(*)>(_a[3]))); break;
        case 26: _t->unregisterFloatStreamingHostBuffers(); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QOffscreenSurface* >(); break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QOpenGLContext* >(); break;
            case 2:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QThread* >(); break;
            }
            break;
        case 14:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< AcquisitionSystem* >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (Processing::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::initializationDone)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (Processing::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::initializationFailed)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (Processing::*)(QOpenGLContext * , QOffscreenSurface * , QThread * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::initOpenGL)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (Processing::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::initOpenGLenFaceView)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (Processing::*)(OctAlgorithmParameters::RecordingParams );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::initRawRecorder)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (Processing::*)(OctAlgorithmParameters::RecordingParams );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::initProcessedRecorder)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (Processing::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::processingDone)) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (Processing::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::streamingBufferEnabled)) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (Processing::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::processedRecordDone)) {
                *result = 8;
                return;
            }
        }
        {
            using _t = void (Processing::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::rawRecordDone)) {
                *result = 9;
                return;
            }
        }
        {
            using _t = void (Processing::*)(void * , unsigned  , unsigned int , unsigned int , unsigned int , unsigned int , unsigned int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::rawData)) {
                *result = 10;
                return;
            }
        }
        {
            using _t = void (Processing::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::info)) {
                *result = 11;
                return;
            }
        }
        {
            using _t = void (Processing::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::error)) {
                *result = 12;
                return;
            }
        }
        {
            using _t = void (Processing::*)(QString , QString , QString , QString , QString , QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Processing::updateInfoBox)) {
                *result = 13;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject Processing::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_Processing.data,
    qt_meta_data_Processing,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *Processing::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Processing::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_Processing.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int Processing::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 27)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 27;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 27)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 27;
    }
    return _id;
}

// SIGNAL 0
void Processing::initializationDone()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void Processing::initializationFailed()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void Processing::initOpenGL(QOpenGLContext * _t1, QOffscreenSurface * _t2, QThread * _t3)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void Processing::initOpenGLenFaceView()
{
    QMetaObject::activate(this, &staticMetaObject, 3, nullptr);
}

// SIGNAL 4
void Processing::initRawRecorder(OctAlgorithmParameters::RecordingParams _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void Processing::initProcessedRecorder(OctAlgorithmParameters::RecordingParams _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void Processing::processingDone()
{
    QMetaObject::activate(this, &staticMetaObject, 6, nullptr);
}

// SIGNAL 7
void Processing::streamingBufferEnabled(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 7, _a);
}

// SIGNAL 8
void Processing::processedRecordDone()
{
    QMetaObject::activate(this, &staticMetaObject, 8, nullptr);
}

// SIGNAL 9
void Processing::rawRecordDone()
{
    QMetaObject::activate(this, &staticMetaObject, 9, nullptr);
}

// SIGNAL 10
void Processing::rawData(void * _t1, unsigned  _t2, unsigned int _t3, unsigned int _t4, unsigned int _t5, unsigned int _t6, unsigned int _t7)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t5))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t6))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t7))) };
    QMetaObject::activate(this, &staticMetaObject, 10, _a);
}

// SIGNAL 11
void Processing::info(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 11, _a);
}

// SIGNAL 12
void Processing::error(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 12, _a);
}

// SIGNAL 13
void Processing::updateInfoBox(QString _t1, QString _t2, QString _t3, QString _t4, QString _t5, QString _t6)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t5))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t6))) };
    QMetaObject::activate(this, &staticMetaObject, 13, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
