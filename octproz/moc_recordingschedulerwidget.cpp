/****************************************************************************
** Meta object code from reading C++ file 'recordingschedulerwidget.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../projects/OCTproZ/src/octproz_project/octproz/src/recordingschedulerwidget.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'recordingschedulerwidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_RecordingSchedulerWidget_t {
    QByteArrayData data[18];
    char stringdata0[256];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_RecordingSchedulerWidget_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_RecordingSchedulerWidget_t qt_meta_stringdata_RecordingSchedulerWidget = {
    {
QT_MOC_LITERAL(0, 0, 24), // "RecordingSchedulerWidget"
QT_MOC_LITERAL(1, 25, 4), // "info"
QT_MOC_LITERAL(2, 30, 0), // ""
QT_MOC_LITERAL(3, 31, 11), // "infoMessage"
QT_MOC_LITERAL(4, 43, 5), // "error"
QT_MOC_LITERAL(5, 49, 12), // "errorMessage"
QT_MOC_LITERAL(6, 62, 15), // "scheduleStarted"
QT_MOC_LITERAL(7, 78, 15), // "scheduleStopped"
QT_MOC_LITERAL(8, 94, 15), // "progressUpdated"
QT_MOC_LITERAL(9, 110, 9), // "completed"
QT_MOC_LITERAL(10, 120, 5), // "total"
QT_MOC_LITERAL(11, 126, 20), // "timeUntilNextUpdated"
QT_MOC_LITERAL(12, 147, 7), // "seconds"
QT_MOC_LITERAL(13, 155, 17), // "scheduleCompleted"
QT_MOC_LITERAL(14, 173, 13), // "startSchedule"
QT_MOC_LITERAL(15, 187, 12), // "stopSchedule"
QT_MOC_LITERAL(16, 200, 28), // "updateStartDelaySecondsValue"
QT_MOC_LITERAL(17, 229, 26) // "updateIntervalSecondsValue"

    },
    "RecordingSchedulerWidget\0info\0\0"
    "infoMessage\0error\0errorMessage\0"
    "scheduleStarted\0scheduleStopped\0"
    "progressUpdated\0completed\0total\0"
    "timeUntilNextUpdated\0seconds\0"
    "scheduleCompleted\0startSchedule\0"
    "stopSchedule\0updateStartDelaySecondsValue\0"
    "updateIntervalSecondsValue"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_RecordingSchedulerWidget[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      11,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   69,    2, 0x06 /* Public */,
       4,    1,   72,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       6,    0,   75,    2, 0x0a /* Public */,
       7,    0,   76,    2, 0x0a /* Public */,
       8,    2,   77,    2, 0x0a /* Public */,
      11,    1,   82,    2, 0x0a /* Public */,
      13,    0,   85,    2, 0x0a /* Public */,
      14,    0,   86,    2, 0x08 /* Private */,
      15,    0,   87,    2, 0x08 /* Private */,
      16,    0,   88,    2, 0x08 /* Private */,
      17,    0,   89,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString,    3,
    QMetaType::Void, QMetaType::QString,    5,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Int,    9,   10,
    QMetaType::Void, QMetaType::Int,   12,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void RecordingSchedulerWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<RecordingSchedulerWidget *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->info((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 1: _t->error((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 2: _t->scheduleStarted(); break;
        case 3: _t->scheduleStopped(); break;
        case 4: _t->progressUpdated((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 5: _t->timeUntilNextUpdated((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->scheduleCompleted(); break;
        case 7: _t->startSchedule(); break;
        case 8: _t->stopSchedule(); break;
        case 9: _t->updateStartDelaySecondsValue(); break;
        case 10: _t->updateIntervalSecondsValue(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (RecordingSchedulerWidget::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RecordingSchedulerWidget::info)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (RecordingSchedulerWidget::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&RecordingSchedulerWidget::error)) {
                *result = 1;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject RecordingSchedulerWidget::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_RecordingSchedulerWidget.data,
    qt_meta_data_RecordingSchedulerWidget,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *RecordingSchedulerWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *RecordingSchedulerWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_RecordingSchedulerWidget.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int RecordingSchedulerWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 11)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 11;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 11)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 11;
    }
    return _id;
}

// SIGNAL 0
void RecordingSchedulerWidget::info(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void RecordingSchedulerWidget::error(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
