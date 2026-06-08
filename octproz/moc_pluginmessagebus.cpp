/****************************************************************************
** Meta object code from reading C++ file 'pluginmessagebus.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../projects/OCTproZ/src/octproz_project/octproz/src/pluginmessagebus.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'pluginmessagebus.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_PluginMessageBus_t {
    QByteArrayData data[8];
    char stringdata0[82];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PluginMessageBus_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PluginMessageBus_t qt_meta_stringdata_PluginMessageBus = {
    {
QT_MOC_LITERAL(0, 0, 16), // "PluginMessageBus"
QT_MOC_LITERAL(1, 17, 11), // "commandSent"
QT_MOC_LITERAL(2, 29, 0), // ""
QT_MOC_LITERAL(3, 30, 6), // "sender"
QT_MOC_LITERAL(4, 37, 12), // "targetPlugin"
QT_MOC_LITERAL(5, 50, 7), // "command"
QT_MOC_LITERAL(6, 58, 6), // "params"
QT_MOC_LITERAL(7, 65, 16) // "commandBroadcast"

    },
    "PluginMessageBus\0commandSent\0\0sender\0"
    "targetPlugin\0command\0params\0"
    "commandBroadcast"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PluginMessageBus[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    4,   24,    2, 0x06 /* Public */,
       7,    3,   33,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString, QMetaType::QString, QMetaType::QString, QMetaType::QVariantMap,    3,    4,    5,    6,
    QMetaType::Void, QMetaType::QString, QMetaType::QString, QMetaType::QVariantMap,    3,    5,    6,

       0        // eod
};

void PluginMessageBus::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<PluginMessageBus *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->commandSent((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2])),(*reinterpret_cast< const QString(*)>(_a[3])),(*reinterpret_cast< const QVariantMap(*)>(_a[4]))); break;
        case 1: _t->commandBroadcast((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2])),(*reinterpret_cast< const QVariantMap(*)>(_a[3]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (PluginMessageBus::*)(const QString & , const QString & , const QString & , const QVariantMap & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PluginMessageBus::commandSent)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (PluginMessageBus::*)(const QString & , const QString & , const QVariantMap & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PluginMessageBus::commandBroadcast)) {
                *result = 1;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject PluginMessageBus::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_PluginMessageBus.data,
    qt_meta_data_PluginMessageBus,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *PluginMessageBus::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PluginMessageBus::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_PluginMessageBus.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int PluginMessageBus::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 2)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void PluginMessageBus::commandSent(const QString & _t1, const QString & _t2, const QString & _t3, const QVariantMap & _t4)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void PluginMessageBus::commandBroadcast(const QString & _t1, const QString & _t2, const QVariantMap & _t3)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
