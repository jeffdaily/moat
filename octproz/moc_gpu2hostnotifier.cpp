/****************************************************************************
** Meta object code from reading C++ file 'gpu2hostnotifier.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../projects/OCTproZ/src/octproz_project/octproz/src/gpu2hostnotifier.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'gpu2hostnotifier.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_Gpu2HostNotifier_t {
    QByteArrayData data[28];
    char stringdata0[526];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Gpu2HostNotifier_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Gpu2HostNotifier_t qt_meta_stringdata_Gpu2HostNotifier = {
    {
QT_MOC_LITERAL(0, 0, 16), // "Gpu2HostNotifier"
QT_MOC_LITERAL(1, 17, 19), // "processedRecordDone"
QT_MOC_LITERAL(2, 37, 0), // ""
QT_MOC_LITERAL(3, 38, 12), // "recordBuffer"
QT_MOC_LITERAL(4, 51, 19), // "newGpuDataAvailable"
QT_MOC_LITERAL(5, 71, 9), // "rawBuffer"
QT_MOC_LITERAL(6, 81, 8), // "bitDepth"
QT_MOC_LITERAL(7, 90, 14), // "samplesPerLine"
QT_MOC_LITERAL(8, 105, 13), // "linesPerFrame"
QT_MOC_LITERAL(9, 119, 15), // "framesPerBuffer"
QT_MOC_LITERAL(10, 135, 16), // "buffersPerVolume"
QT_MOC_LITERAL(11, 152, 15), // "currentBufferNr"
QT_MOC_LITERAL(12, 168, 24), // "newGpuFloatDataAvailable"
QT_MOC_LITERAL(13, 193, 4), // "data"
QT_MOC_LITERAL(14, 198, 14), // "ascansPerBscan"
QT_MOC_LITERAL(15, 213, 15), // "bscansPerBuffer"
QT_MOC_LITERAL(16, 229, 18), // "backgroundRecorded"
QT_MOC_LITERAL(17, 248, 23), // "bscanDisplayBufferReady"
QT_MOC_LITERAL(18, 272, 24), // "enfaceDisplayBufferReady"
QT_MOC_LITERAL(19, 297, 24), // "volumeDisplayBufferReady"
QT_MOC_LITERAL(20, 322, 26), // "emitCurrentStreamingBuffer"
QT_MOC_LITERAL(21, 349, 19), // "currStreamingBuffer"
QT_MOC_LITERAL(22, 369, 31), // "emitCurrentFloatStreamingBuffer"
QT_MOC_LITERAL(23, 401, 15), // "streamingBuffer"
QT_MOC_LITERAL(24, 417, 22), // "emitBackgroundRecorded"
QT_MOC_LITERAL(25, 440, 27), // "emitBscanDisplayBufferReady"
QT_MOC_LITERAL(26, 468, 28), // "emitEnfaceDisplayBufferReady"
QT_MOC_LITERAL(27, 497, 28) // "emitVolumeDisplayBufferReady"

    },
    "Gpu2HostNotifier\0processedRecordDone\0"
    "\0recordBuffer\0newGpuDataAvailable\0"
    "rawBuffer\0bitDepth\0samplesPerLine\0"
    "linesPerFrame\0framesPerBuffer\0"
    "buffersPerVolume\0currentBufferNr\0"
    "newGpuFloatDataAvailable\0data\0"
    "ascansPerBscan\0bscansPerBuffer\0"
    "backgroundRecorded\0bscanDisplayBufferReady\0"
    "enfaceDisplayBufferReady\0"
    "volumeDisplayBufferReady\0"
    "emitCurrentStreamingBuffer\0"
    "currStreamingBuffer\0emitCurrentFloatStreamingBuffer\0"
    "streamingBuffer\0emitBackgroundRecorded\0"
    "emitBscanDisplayBufferReady\0"
    "emitEnfaceDisplayBufferReady\0"
    "emitVolumeDisplayBufferReady"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Gpu2HostNotifier[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      13,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       7,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   79,    2, 0x06 /* Public */,
       4,    7,   82,    2, 0x06 /* Public */,
      12,    7,   97,    2, 0x06 /* Public */,
      16,    0,  112,    2, 0x06 /* Public */,
      17,    0,  113,    2, 0x06 /* Public */,
      18,    0,  114,    2, 0x06 /* Public */,
      19,    0,  115,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      20,    1,  116,    2, 0x0a /* Public */,
      22,    1,  119,    2, 0x0a /* Public */,
      24,    0,  122,    2, 0x0a /* Public */,
      25,    1,  123,    2, 0x0a /* Public */,
      26,    1,  126,    2, 0x0a /* Public */,
      27,    1,  129,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::VoidStar,    3,
    QMetaType::Void, QMetaType::VoidStar, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt,    5,    6,    7,    8,    9,   10,   11,
    QMetaType::Void, QMetaType::VoidStar, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt,   13,    6,    7,   14,   15,   10,   11,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, QMetaType::VoidStar,   21,
    QMetaType::Void, QMetaType::VoidStar,   23,
    QMetaType::Void,
    QMetaType::Void, QMetaType::VoidStar,   13,
    QMetaType::Void, QMetaType::VoidStar,   13,
    QMetaType::Void, QMetaType::VoidStar,   13,

       0        // eod
};

void Gpu2HostNotifier::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<Gpu2HostNotifier *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->processedRecordDone((*reinterpret_cast< void*(*)>(_a[1]))); break;
        case 1: _t->newGpuDataAvailable((*reinterpret_cast< void*(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< uint(*)>(_a[3])),(*reinterpret_cast< uint(*)>(_a[4])),(*reinterpret_cast< uint(*)>(_a[5])),(*reinterpret_cast< uint(*)>(_a[6])),(*reinterpret_cast< uint(*)>(_a[7]))); break;
        case 2: _t->newGpuFloatDataAvailable((*reinterpret_cast< void*(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< uint(*)>(_a[3])),(*reinterpret_cast< uint(*)>(_a[4])),(*reinterpret_cast< uint(*)>(_a[5])),(*reinterpret_cast< uint(*)>(_a[6])),(*reinterpret_cast< uint(*)>(_a[7]))); break;
        case 3: _t->backgroundRecorded(); break;
        case 4: _t->bscanDisplayBufferReady(); break;
        case 5: _t->enfaceDisplayBufferReady(); break;
        case 6: _t->volumeDisplayBufferReady(); break;
        case 7: _t->emitCurrentStreamingBuffer((*reinterpret_cast< void*(*)>(_a[1]))); break;
        case 8: _t->emitCurrentFloatStreamingBuffer((*reinterpret_cast< void*(*)>(_a[1]))); break;
        case 9: _t->emitBackgroundRecorded(); break;
        case 10: _t->emitBscanDisplayBufferReady((*reinterpret_cast< void*(*)>(_a[1]))); break;
        case 11: _t->emitEnfaceDisplayBufferReady((*reinterpret_cast< void*(*)>(_a[1]))); break;
        case 12: _t->emitVolumeDisplayBufferReady((*reinterpret_cast< void*(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (Gpu2HostNotifier::*)(void * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Gpu2HostNotifier::processedRecordDone)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (Gpu2HostNotifier::*)(void * , unsigned int , unsigned int , unsigned int , unsigned int , unsigned int , unsigned int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Gpu2HostNotifier::newGpuDataAvailable)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (Gpu2HostNotifier::*)(void * , unsigned int , unsigned int , unsigned int , unsigned int , unsigned int , unsigned int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Gpu2HostNotifier::newGpuFloatDataAvailable)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (Gpu2HostNotifier::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Gpu2HostNotifier::backgroundRecorded)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (Gpu2HostNotifier::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Gpu2HostNotifier::bscanDisplayBufferReady)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (Gpu2HostNotifier::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Gpu2HostNotifier::enfaceDisplayBufferReady)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (Gpu2HostNotifier::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Gpu2HostNotifier::volumeDisplayBufferReady)) {
                *result = 6;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject Gpu2HostNotifier::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_Gpu2HostNotifier.data,
    qt_meta_data_Gpu2HostNotifier,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *Gpu2HostNotifier::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Gpu2HostNotifier::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_Gpu2HostNotifier.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int Gpu2HostNotifier::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 13)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 13;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 13)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 13;
    }
    return _id;
}

// SIGNAL 0
void Gpu2HostNotifier::processedRecordDone(void * _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void Gpu2HostNotifier::newGpuDataAvailable(void * _t1, unsigned int _t2, unsigned int _t3, unsigned int _t4, unsigned int _t5, unsigned int _t6, unsigned int _t7)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t5))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t6))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t7))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void Gpu2HostNotifier::newGpuFloatDataAvailable(void * _t1, unsigned int _t2, unsigned int _t3, unsigned int _t4, unsigned int _t5, unsigned int _t6, unsigned int _t7)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t5))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t6))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t7))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void Gpu2HostNotifier::backgroundRecorded()
{
    QMetaObject::activate(this, &staticMetaObject, 3, nullptr);
}

// SIGNAL 4
void Gpu2HostNotifier::bscanDisplayBufferReady()
{
    QMetaObject::activate(this, &staticMetaObject, 4, nullptr);
}

// SIGNAL 5
void Gpu2HostNotifier::enfaceDisplayBufferReady()
{
    QMetaObject::activate(this, &staticMetaObject, 5, nullptr);
}

// SIGNAL 6
void Gpu2HostNotifier::volumeDisplayBufferReady()
{
    QMetaObject::activate(this, &staticMetaObject, 6, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
