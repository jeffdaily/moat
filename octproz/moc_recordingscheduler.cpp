/****************************************************************************
** Meta object code from reading C++ file 'recordingscheduler.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../projects/OCTproZ/src/octproz_project/octproz/src/recordingscheduler.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'recordingscheduler.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_RecordingScheduler_t {
    QByteArrayData data[25];
    char stringdata0[369];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RecordingScheduler_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RecordingScheduler_t qt_meta_stringdata_RecordingScheduler = {
    {
QT_MOC_LITERAL(0, 0, 18), // "RecordingScheduler"
QT_MOC_LITERAL(1, 19, 15), // "scheduleStarted"
QT_MOC_LITERAL(2, 35, 0), // ""
QT_MOC_LITERAL(3, 36, 15), // "scheduleStopped"
QT_MOC_LITERAL(4, 52, 18), // "recordingTriggered"
QT_MOC_LITERAL(5, 71, 15), // "progressUpdated"
QT_MOC_LITERAL(6, 87, 19), // "recordingsCompleted"
QT_MOC_LITERAL(7, 107, 15), // "totalRecordings"
QT_MOC_LITERAL(8, 123, 29), // "timeUntilNextRecordingUpdated"
QT_MOC_LITERAL(9, 153, 7), // "seconds"
QT_MOC_LITERAL(10, 161, 17), // "scheduleCompleted"
QT_MOC_LITERAL(11, 179, 13), // "delayOccurred"
QT_MOC_LITERAL(12, 193, 12), // "delaySeconds"
QT_MOC_LITERAL(13, 206, 4), // "info"
QT_MOC_LITERAL(14, 211, 11), // "infoMessage"
QT_MOC_LITERAL(15, 223, 5), // "error"
QT_MOC_LITERAL(16, 229, 12), // "errorMessage"
QT_MOC_LITERAL(17, 242, 13), // "startSchedule"
QT_MOC_LITERAL(18, 256, 17), // "startDelaySeconds"
QT_MOC_LITERAL(19, 274, 15), // "intervalSeconds"
QT_MOC_LITERAL(20, 290, 12), // "stopSchedule"
QT_MOC_LITERAL(21, 303, 17), // "recordingFinished"
QT_MOC_LITERAL(22, 321, 18), // "startDelayFinished"
QT_MOC_LITERAL(23, 340, 8), // "interval"
QT_MOC_LITERAL(24, 349, 19) // "updateTimeUntilNext"

    },
    "RecordingScheduler\0scheduleStarted\0\0"
    "scheduleStopped\0recordingTriggered\0"
    "progressUpdated\0recordingsCompleted\0"
    "totalRecordings\0timeUntilNextRecordingUpdated\0"
    "seconds\0scheduleCompleted\0delayOccurred\0"
    "delaySeconds\0info\0infoMessage\0error\0"
    "errorMessage\0startSchedule\0startDelaySeconds\0"
    "intervalSeconds\0stopSchedule\0"
    "recordingFinished\0startDelayFinished\0"
    "interval\0updateTimeUntilNext"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RecordingScheduler[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      15,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       9,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   89,    2, 0x06 /* Public */,
       3,    0,   90,    2, 0x06 /* Public */,
       4,    0,   91,    2, 0x06 /* Public */,
       5,    2,   92,    2, 0x06 /* Public */,
       8,    1,   97,    2, 0x06 /* Public */,
      10,    0,  100,    2, 0x06 /* Public */,
      11,    1,  101,    2, 0x06 /* Public */,
      13,    1,  104,    2, 0x06 /* Public */,
      15,    1,  107,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      17,    3,  110,    2, 0x0a /* Public */,
      20,    0,  117,    2, 0x0a /* Public */,
      21,    0,  118,    2, 0x0a /* Public */,
      22,    0,  119,    2, 0x08 /* Private */,
      23,    0,  120,    2, 0x08 /* Private */,
      24,    0,  121,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    6,    7,
    QMetaType::Void, QMetaType::Int,    9,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   12,
    QMetaType::Void, QMetaType::QString,   14,
    QMetaType::Void, QMetaType::QString,   16,

 // slots: parameters
    QMetaType::Void, QMetaType::Int, QMetaType::Int, QMetaType::Int,   18,   19,    7,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void RecordingScheduler::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<RecordingScheduler *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->scheduleStarted(); break;
        case 1: _t->scheduleStopped(); break;
        case 2: _t->recordingTriggered(); break;
        case 3: _t->progressUpdated((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 4: _t->timeUntilNextRecordingUpdated((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->scheduleCompleted(); break;
        case 6: _t->delayOccurred((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->info((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 8: _t->error((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 9: _t->startSchedule((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 10: _t->stopSchedule(); break;
        case 11: _t->recordingFinished(); break;
        case 12: _t->startDelayFinished(); break;
        case 13: _t->interval(); break;
        case 14: _t->updateTimeUntilNext(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (RecordingScheduler::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RecordingScheduler::scheduleStarted)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (RecordingScheduler::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RecordingScheduler::scheduleStopped)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (RecordingScheduler::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RecordingScheduler::recordingTriggered)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (RecordingScheduler::*)(int , int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RecordingScheduler::progressUpdated)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (RecordingScheduler::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RecordingScheduler::timeUntilNextRecordingUpdated)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (RecordingScheduler::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RecordingScheduler::scheduleCompleted)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (RecordingScheduler::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RecordingScheduler::delayOccurred)) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (RecordingScheduler::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RecordingScheduler::info)) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (RecordingScheduler::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RecordingScheduler::error)) {
                *result = 8;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject RecordingScheduler::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_RecordingScheduler.data,
    qt_meta_data_RecordingScheduler,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *RecordingScheduler::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RecordingScheduler::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_RecordingScheduler.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int RecordingScheduler::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 15)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 15;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 15)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 15;
    }
    return _id;
}

// SIGNAL 0
void RecordingScheduler::scheduleStarted()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void RecordingScheduler::scheduleStopped()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void RecordingScheduler::recordingTriggered()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void RecordingScheduler::progressUpdated(int _t1, int _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void RecordingScheduler::timeUntilNextRecordingUpdated(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void RecordingScheduler::scheduleCompleted()
{
    QMetaObject::activate(this, &staticMetaObject, 5, nullptr);
}

// SIGNAL 6
void RecordingScheduler::delayOccurred(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}

// SIGNAL 7
void RecordingScheduler::info(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 7, _a);
}

// SIGNAL 8
void RecordingScheduler::error(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 8, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
