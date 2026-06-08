/****************************************************************************
** Meta object code from reading C++ file 'virtualoctsystemsettingsdialog.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../projects/OCTproZ/src/octproz_project/octproz_plugins/octproz-virtual-oct-system/src/virtualoctsystemsettingsdialog.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'virtualoctsystemsettingsdialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_VirtualOCTSystemSettingsDialog_t {
    QByteArrayData data[10];
    char stringdata0[144];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_VirtualOCTSystemSettingsDialog_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_VirtualOCTSystemSettingsDialog_t qt_meta_stringdata_VirtualOCTSystemSettingsDialog = {
    {
QT_MOC_LITERAL(0, 0, 30), // "VirtualOCTSystemSettingsDialog"
QT_MOC_LITERAL(1, 31, 15), // "settingsUpdated"
QT_MOC_LITERAL(2, 47, 0), // ""
QT_MOC_LITERAL(3, 48, 15), // "simulatorParams"
QT_MOC_LITERAL(4, 64, 9), // "newParams"
QT_MOC_LITERAL(5, 74, 15), // "slot_selectFile"
QT_MOC_LITERAL(6, 90, 10), // "slot_apply"
QT_MOC_LITERAL(7, 101, 14), // "slot_enableGui"
QT_MOC_LITERAL(8, 116, 6), // "enable"
QT_MOC_LITERAL(9, 123, 20) // "slot_checkWidthValue"

    },
    "VirtualOCTSystemSettingsDialog\0"
    "settingsUpdated\0\0simulatorParams\0"
    "newParams\0slot_selectFile\0slot_apply\0"
    "slot_enableGui\0enable\0slot_checkWidthValue"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_VirtualOCTSystemSettingsDialog[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   39,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       5,    0,   42,    2, 0x0a /* Public */,
       6,    0,   43,    2, 0x0a /* Public */,
       7,    1,   44,    2, 0x0a /* Public */,
       9,    0,   47,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    4,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    8,
    QMetaType::Void,

       0        // eod
};

void VirtualOCTSystemSettingsDialog::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<VirtualOCTSystemSettingsDialog *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->settingsUpdated((*reinterpret_cast< simulatorParams(*)>(_a[1]))); break;
        case 1: _t->slot_selectFile(); break;
        case 2: _t->slot_apply(); break;
        case 3: _t->slot_enableGui((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 4: _t->slot_checkWidthValue(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (VirtualOCTSystemSettingsDialog::*)(simulatorParams );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&VirtualOCTSystemSettingsDialog::settingsUpdated)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject VirtualOCTSystemSettingsDialog::staticMetaObject = { {
    QMetaObject::SuperData::link<QDialog::staticMetaObject>(),
    qt_meta_stringdata_VirtualOCTSystemSettingsDialog.data,
    qt_meta_data_VirtualOCTSystemSettingsDialog,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *VirtualOCTSystemSettingsDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *VirtualOCTSystemSettingsDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_VirtualOCTSystemSettingsDialog.stringdata0))
        return static_cast<void*>(this);
    return QDialog::qt_metacast(_clname);
}

int VirtualOCTSystemSettingsDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void VirtualOCTSystemSettingsDialog::settingsUpdated(simulatorParams _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
