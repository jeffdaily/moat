/****************************************************************************
** Meta object code from reading C++ file 'virtualoctsystem.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../projects/OCTproZ/src/octproz_project/octproz_plugins/octproz-virtual-oct-system/src/virtualoctsystem.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#include <QtCore/qplugin.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'virtualoctsystem.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_VirtualOCTSystem_t {
    QByteArrayData data[7];
    char stringdata0[79];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_VirtualOCTSystem_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_VirtualOCTSystem_t qt_meta_stringdata_VirtualOCTSystem = {
    {
QT_MOC_LITERAL(0, 0, 16), // "VirtualOCTSystem"
QT_MOC_LITERAL(1, 17, 9), // "enableGui"
QT_MOC_LITERAL(2, 27, 0), // ""
QT_MOC_LITERAL(3, 28, 6), // "enable"
QT_MOC_LITERAL(4, 35, 17), // "slot_updateParams"
QT_MOC_LITERAL(5, 53, 15), // "simulatorParams"
QT_MOC_LITERAL(6, 69, 9) // "newParams"

    },
    "VirtualOCTSystem\0enableGui\0\0enable\0"
    "slot_updateParams\0simulatorParams\0"
    "newParams"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_VirtualOCTSystem[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   24,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       4,    1,   27,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Bool,    3,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 5,    6,

       0        // eod
};

void VirtualOCTSystem::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<VirtualOCTSystem *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->enableGui((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->slot_updateParams((*reinterpret_cast< simulatorParams(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (VirtualOCTSystem::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&VirtualOCTSystem::enableGui)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject VirtualOCTSystem::staticMetaObject = { {
    QMetaObject::SuperData::link<AcquisitionSystem::staticMetaObject>(),
    qt_meta_stringdata_VirtualOCTSystem.data,
    qt_meta_data_VirtualOCTSystem,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *VirtualOCTSystem::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *VirtualOCTSystem::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_VirtualOCTSystem.stringdata0))
        return static_cast<void*>(this);
    if (!strcmp(_clname, "octproz.acquisition.interface"))
        return static_cast< AcquisitionSystem*>(this);
    if (!strcmp(_clname, "octproz.plugin.interface"))
        return static_cast< Plugin*>(this);
    return AcquisitionSystem::qt_metacast(_clname);
}

int VirtualOCTSystem::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AcquisitionSystem::qt_metacall(_c, _id, _a);
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
void VirtualOCTSystem::enableGui(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

QT_PLUGIN_METADATA_SECTION
static constexpr unsigned char qt_pluginMetaData[] = {
    'Q', 'T', 'M', 'E', 'T', 'A', 'D', 'A', 'T', 'A', ' ', '!',
    // metadata version, Qt version, architectural requirements
    0, QT_VERSION_MAJOR, QT_VERSION_MINOR, qPluginArchRequirements(),
    0xbf, 
    // "IID"
    0x02,  0x78,  0x1d,  'o',  'c',  't',  'p',  'r', 
    'o',  'z',  '.',  'a',  'c',  'q',  'u',  'i', 
    's',  'i',  't',  'i',  'o',  'n',  '.',  'i', 
    'n',  't',  'e',  'r',  'f',  'a',  'c',  'e', 
    // "className"
    0x03,  0x70,  'V',  'i',  'r',  't',  'u',  'a', 
    'l',  'O',  'C',  'T',  'S',  'y',  's',  't', 
    'e',  'm', 
    0xff, 
};
QT_MOC_EXPORT_PLUGIN(VirtualOCTSystem, VirtualOCTSystem)

QT_WARNING_POP
QT_END_MOC_NAMESPACE
