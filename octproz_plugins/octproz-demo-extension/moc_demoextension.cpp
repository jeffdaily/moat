/****************************************************************************
** Meta object code from reading C++ file 'demoextension.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../projects/OCTproZ/src/octproz_project/octproz_plugins/octproz-demo-extension/src/demoextension.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#include <QtCore/qplugin.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'demoextension.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_DemoExtension_t {
    QByteArrayData data[14];
    char stringdata0[179];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_DemoExtension_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_DemoExtension_t qt_meta_stringdata_DemoExtension = {
    {
QT_MOC_LITERAL(0, 0, 13), // "DemoExtension"
QT_MOC_LITERAL(1, 14, 13), // "setParameters"
QT_MOC_LITERAL(2, 28, 0), // ""
QT_MOC_LITERAL(3, 29, 10), // "demoParams"
QT_MOC_LITERAL(4, 40, 6), // "params"
QT_MOC_LITERAL(5, 47, 15), // "rawDataReceived"
QT_MOC_LITERAL(6, 63, 6), // "buffer"
QT_MOC_LITERAL(7, 70, 8), // "bitDepth"
QT_MOC_LITERAL(8, 79, 14), // "samplesPerLine"
QT_MOC_LITERAL(9, 94, 13), // "linesPerFrame"
QT_MOC_LITERAL(10, 108, 15), // "framesPerBuffer"
QT_MOC_LITERAL(11, 124, 16), // "buffersPerVolume"
QT_MOC_LITERAL(12, 141, 15), // "currentBufferNr"
QT_MOC_LITERAL(13, 157, 21) // "processedDataReceived"

    },
    "DemoExtension\0setParameters\0\0demoParams\0"
    "params\0rawDataReceived\0buffer\0bitDepth\0"
    "samplesPerLine\0linesPerFrame\0"
    "framesPerBuffer\0buffersPerVolume\0"
    "currentBufferNr\0processedDataReceived"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_DemoExtension[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   29,    2, 0x0a /* Public */,
       5,    7,   32,    2, 0x0a /* Public */,
      13,    7,   47,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void, QMetaType::VoidStar, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt,    6,    7,    8,    9,   10,   11,   12,
    QMetaType::Void, QMetaType::VoidStar, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt, QMetaType::UInt,    6,    7,    8,    9,   10,   11,   12,

       0        // eod
};

void DemoExtension::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<DemoExtension *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->setParameters((*reinterpret_cast< demoParams(*)>(_a[1]))); break;
        case 1: _t->rawDataReceived((*reinterpret_cast< void*(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< uint(*)>(_a[3])),(*reinterpret_cast< uint(*)>(_a[4])),(*reinterpret_cast< uint(*)>(_a[5])),(*reinterpret_cast< uint(*)>(_a[6])),(*reinterpret_cast< uint(*)>(_a[7]))); break;
        case 2: _t->processedDataReceived((*reinterpret_cast< void*(*)>(_a[1])),(*reinterpret_cast< uint(*)>(_a[2])),(*reinterpret_cast< uint(*)>(_a[3])),(*reinterpret_cast< uint(*)>(_a[4])),(*reinterpret_cast< uint(*)>(_a[5])),(*reinterpret_cast< uint(*)>(_a[6])),(*reinterpret_cast< uint(*)>(_a[7]))); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject DemoExtension::staticMetaObject = { {
    QMetaObject::SuperData::link<Extension::staticMetaObject>(),
    qt_meta_stringdata_DemoExtension.data,
    qt_meta_data_DemoExtension,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *DemoExtension::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *DemoExtension::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_DemoExtension.stringdata0))
        return static_cast<void*>(this);
    if (!strcmp(_clname, "octproz.extension.interface"))
        return static_cast< Extension*>(this);
    if (!strcmp(_clname, "octproz.plugin.interface"))
        return static_cast< Plugin*>(this);
    return Extension::qt_metacast(_clname);
}

int DemoExtension::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = Extension::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 3)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 3;
    }
    return _id;
}

QT_PLUGIN_METADATA_SECTION
static constexpr unsigned char qt_pluginMetaData[] = {
    'Q', 'T', 'M', 'E', 'T', 'A', 'D', 'A', 'T', 'A', ' ', '!',
    // metadata version, Qt version, architectural requirements
    0, QT_VERSION_MAJOR, QT_VERSION_MINOR, qPluginArchRequirements(),
    0xbf, 
    // "IID"
    0x02,  0x78,  0x1b,  'o',  'c',  't',  'p',  'r', 
    'o',  'z',  '.',  'e',  'x',  't',  'e',  'n', 
    's',  'i',  'o',  'n',  '.',  'i',  'n',  't', 
    'e',  'r',  'f',  'a',  'c',  'e', 
    // "className"
    0x03,  0x6d,  'D',  'e',  'm',  'o',  'E',  'x', 
    't',  'e',  'n',  's',  'i',  'o',  'n', 
    0xff, 
};
QT_MOC_EXPORT_PLUGIN(DemoExtension, DemoExtension)

QT_WARNING_POP
QT_END_MOC_NAMESPACE
