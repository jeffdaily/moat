/****************************************************************************
** Meta object code from reading C++ file 'controlpanel.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../projects/OCTproZ/src/octproz_project/octproz/src/controlpanel.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'controlpanel.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ControlPanel3D_t {
    QByteArrayData data[16];
    char stringdata0[206];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ControlPanel3D_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ControlPanel3D_t qt_meta_stringdata_ControlPanel3D = {
    {
QT_MOC_LITERAL(0, 0, 14), // "ControlPanel3D"
QT_MOC_LITERAL(1, 15, 24), // "displayParametersChanged"
QT_MOC_LITERAL(2, 40, 0), // ""
QT_MOC_LITERAL(3, 41, 16), // "GLWindow3DParams"
QT_MOC_LITERAL(4, 58, 6), // "params"
QT_MOC_LITERAL(5, 65, 15), // "settingsChanged"
QT_MOC_LITERAL(6, 81, 11), // "lutSelected"
QT_MOC_LITERAL(7, 93, 3), // "lut"
QT_MOC_LITERAL(8, 97, 17), // "dialogAboutToOpen"
QT_MOC_LITERAL(9, 115, 12), // "dialogClosed"
QT_MOC_LITERAL(10, 128, 5), // "error"
QT_MOC_LITERAL(11, 134, 4), // "info"
QT_MOC_LITERAL(12, 139, 23), // "updateDisplayParameters"
QT_MOC_LITERAL(13, 163, 18), // "toggleExtendedView"
QT_MOC_LITERAL(14, 182, 9), // "setParams"
QT_MOC_LITERAL(15, 192, 13) // "openLUTDialog"

    },
    "ControlPanel3D\0displayParametersChanged\0"
    "\0GLWindow3DParams\0params\0settingsChanged\0"
    "lutSelected\0lut\0dialogAboutToOpen\0"
    "dialogClosed\0error\0info\0updateDisplayParameters\0"
    "toggleExtendedView\0setParams\0openLUTDialog"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ControlPanel3D[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      11,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       7,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   69,    2, 0x06 /* Public */,
       5,    0,   72,    2, 0x06 /* Public */,
       6,    1,   73,    2, 0x06 /* Public */,
       8,    0,   76,    2, 0x06 /* Public */,
       9,    0,   77,    2, 0x06 /* Public */,
      10,    1,   78,    2, 0x06 /* Public */,
      11,    1,   81,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      12,    0,   84,    2, 0x0a /* Public */,
      13,    0,   85,    2, 0x0a /* Public */,
      14,    1,   86,    2, 0x0a /* Public */,
      15,    0,   89,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QImage,    7,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,    2,
    QMetaType::Void, QMetaType::QString,    2,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 3,    4,
    QMetaType::Void,

       0        // eod
};

void ControlPanel3D::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<ControlPanel3D *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->displayParametersChanged((*reinterpret_cast< GLWindow3DParams(*)>(_a[1]))); break;
        case 1: _t->settingsChanged(); break;
        case 2: _t->lutSelected((*reinterpret_cast< QImage(*)>(_a[1]))); break;
        case 3: _t->dialogAboutToOpen(); break;
        case 4: _t->dialogClosed(); break;
        case 5: _t->error((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 6: _t->info((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 7: _t->updateDisplayParameters(); break;
        case 8: _t->toggleExtendedView(); break;
        case 9: _t->setParams((*reinterpret_cast< GLWindow3DParams(*)>(_a[1]))); break;
        case 10: _t->openLUTDialog(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ControlPanel3D::*)(GLWindow3DParams );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel3D::displayParametersChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (ControlPanel3D::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel3D::settingsChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (ControlPanel3D::*)(QImage );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel3D::lutSelected)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (ControlPanel3D::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel3D::dialogAboutToOpen)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (ControlPanel3D::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel3D::dialogClosed)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (ControlPanel3D::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel3D::error)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (ControlPanel3D::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ControlPanel3D::info)) {
                *result = 6;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject ControlPanel3D::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_meta_stringdata_ControlPanel3D.data,
    qt_meta_data_ControlPanel3D,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *ControlPanel3D::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ControlPanel3D::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ControlPanel3D.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int ControlPanel3D::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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
void ControlPanel3D::displayParametersChanged(GLWindow3DParams _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void ControlPanel3D::settingsChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void ControlPanel3D::lutSelected(QImage _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void ControlPanel3D::dialogAboutToOpen()
{
    QMetaObject::activate(this, &staticMetaObject, 3, nullptr);
}

// SIGNAL 4
void ControlPanel3D::dialogClosed()
{
    QMetaObject::activate(this, &staticMetaObject, 4, nullptr);
}

// SIGNAL 5
void ControlPanel3D::error(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void ControlPanel3D::info(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
