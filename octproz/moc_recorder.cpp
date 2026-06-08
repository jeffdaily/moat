/****************************************************************************
** Meta object code from reading C++ file 'recorder.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../projects/OCTproZ/src/octproz_project/octproz/src/recorder.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'recorder.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_Recorder_t {
    QByteArrayData data[18];
    char stringdata0[235];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Recorder_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Recorder_t qt_meta_stringdata_Recorder = {
    {
QT_MOC_LITERAL(0, 0, 8), // "Recorder"
QT_MOC_LITERAL(1, 9, 4), // "info"
QT_MOC_LITERAL(2, 14, 0), // ""
QT_MOC_LITERAL(3, 15, 5), // "error"
QT_MOC_LITERAL(4, 21, 13), // "recordingDone"
QT_MOC_LITERAL(5, 35, 13), // "readyToRecord"
QT_MOC_LITERAL(6, 49, 19), // "slot_abortRecording"
QT_MOC_LITERAL(7, 69, 9), // "slot_init"
QT_MOC_LITERAL(8, 79, 39), // "OctAlgorithmParameters::Recor..."
QT_MOC_LITERAL(9, 119, 9), // "recParams"
QT_MOC_LITERAL(10, 129, 11), // "slot_record"
QT_MOC_LITERAL(11, 141, 6), // "buffer"
QT_MOC_LITERAL(12, 148, 8), // "bitDepth"
QT_MOC_LITERAL(13, 157, 14), // "samplesPerLine"
QT_MOC_LITERAL(14, 172, 13), // "linesPerFrame"
QT_MOC_LITERAL(15, 186, 15), // "framesPerBuffer"
QT_MOC_LITERAL(16, 202, 16), // "buffersPerVolume"
QT_MOC_LITERAL(17, 219, 15) // "currentBufferNr"

    },
    "Recorder\0info\0\0error\0recordingDone\0"
    "readyToRecord\0slot_abortRecording\0"
    "slot_init\0OctAlgorithmParameters::RecordingParams\0"
    "recParams\0slot_record\0buffer\0bitDepth\0"
    "samplesPerLine\0linesPerFrame\0"
    "framesPerBuffer\0buffersPerVolume\0"
    "currentBufferNr"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Recorder[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   49,    2, 0x06 /* Public */,
       3,    1,   52,    2, 0x06 /* Public */,
       4,    0,   55,    2, 0x06 /* Public */,
       5,    1,   56,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       6,    0,   59,    2, 0x0a /* Public */,
       7,    1,   60,    2, 0x0a /* Public */,
      10,    7,   63,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString,    1,
    QMetaType::Void, QMetaType::QString,    3,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    2,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 8,    9,
    QMetaType::Void, QMetaType::VoidStar, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt,   11,   12,   13,   14,   15,   16,   17,

       0        // eod
};

void Recorder::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<Recorder *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->info((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 1: _t->error((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 2: _t->recordingDone(); break;
        case 3: _t->readyToRecord((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 4: _t->slot_abortRecording(); break;
        case 5: _t->slot_init((*reinterpret_cast< OctAlgorithmParameters::RecordingParams(*)>(_a[1]))); break;
        case 6: _t->slot_record((*reinterpret_cast< void*(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< uint(*)>(_a[3])),(*reinterpret_cast< uint(*)>(_a[4])),(*reinterpret_cast< uint(*)>(_a[5])),(*reinterpret_cast< uint(*)>(_a[6])),(*reinterpret_cast< uint(*)>(_a[7]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (Recorder::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Recorder::info)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (Recorder::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Recorder::error)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (Recorder::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Recorder::recordingDone)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (Recorder::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Recorder::readyToRecord)) {
                *result = 3;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject Recorder::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_Recorder.data,
    qt_meta_data_Recorder,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *Recorder::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Recorder::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_Recorder.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int Recorder::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void Recorder::info(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void Recorder::error(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void Recorder::recordingDone()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void Recorder::readyToRecord(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
