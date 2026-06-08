/****************************************************************************
** Meta object code from reading C++ file 'plugin.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.13)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../projects/OCTproZ/src/octproz_project/octproz_devkit/src/plugin.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#include <QtCore/QVector>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'plugin.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.13. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_Plugin_t {
    QByteArrayData data[37];
    char stringdata0[470];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Plugin_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Plugin_t qt_meta_stringdata_Plugin = {
    {
QT_MOC_LITERAL(0, 0, 6), // "Plugin"
QT_MOC_LITERAL(1, 7, 4), // "info"
QT_MOC_LITERAL(2, 12, 0), // ""
QT_MOC_LITERAL(3, 13, 5), // "error"
QT_MOC_LITERAL(4, 19, 13), // "storeSettings"
QT_MOC_LITERAL(5, 33, 20), // "setKLinCoeffsRequest"
QT_MOC_LITERAL(6, 54, 7), // "double*"
QT_MOC_LITERAL(7, 62, 2), // "k0"
QT_MOC_LITERAL(8, 65, 2), // "k1"
QT_MOC_LITERAL(9, 68, 2), // "k2"
QT_MOC_LITERAL(10, 71, 2), // "k3"
QT_MOC_LITERAL(11, 74, 24), // "setDispCompCoeffsRequest"
QT_MOC_LITERAL(12, 99, 2), // "d0"
QT_MOC_LITERAL(13, 102, 2), // "d1"
QT_MOC_LITERAL(14, 105, 2), // "d2"
QT_MOC_LITERAL(15, 108, 2), // "d3"
QT_MOC_LITERAL(16, 111, 29), // "setGrayscaleConversionRequest"
QT_MOC_LITERAL(17, 141, 16), // "enableLogScaling"
QT_MOC_LITERAL(18, 158, 3), // "max"
QT_MOC_LITERAL(19, 162, 3), // "min"
QT_MOC_LITERAL(20, 166, 13), // "multiplicator"
QT_MOC_LITERAL(21, 180, 6), // "offset"
QT_MOC_LITERAL(22, 187, 22), // "startProcessingRequest"
QT_MOC_LITERAL(23, 210, 21), // "stopProcessingRequest"
QT_MOC_LITERAL(24, 232, 21), // "startRecordingRequest"
QT_MOC_LITERAL(25, 254, 31), // "setCustomResamplingCurveRequest"
QT_MOC_LITERAL(26, 286, 14), // "QVector<float>"
QT_MOC_LITERAL(27, 301, 11), // "customCurve"
QT_MOC_LITERAL(28, 313, 23), // "loadSettingsFileRequest"
QT_MOC_LITERAL(29, 337, 23), // "saveSettingsFileRequest"
QT_MOC_LITERAL(30, 361, 11), // "sendCommand"
QT_MOC_LITERAL(31, 373, 6), // "sender"
QT_MOC_LITERAL(32, 380, 12), // "targetPlugin"
QT_MOC_LITERAL(33, 393, 7), // "command"
QT_MOC_LITERAL(34, 401, 6), // "params"
QT_MOC_LITERAL(35, 408, 28), // "setKLinCoeffsRequestAccepted"
QT_MOC_LITERAL(36, 437, 32) // "setDispCompCoeffsRequestAccepted"

    },
    "Plugin\0info\0\0error\0storeSettings\0"
    "setKLinCoeffsRequest\0double*\0k0\0k1\0"
    "k2\0k3\0setDispCompCoeffsRequest\0d0\0d1\0"
    "d2\0d3\0setGrayscaleConversionRequest\0"
    "enableLogScaling\0max\0min\0multiplicator\0"
    "offset\0startProcessingRequest\0"
    "stopProcessingRequest\0startRecordingRequest\0"
    "setCustomResamplingCurveRequest\0"
    "QVector<float>\0customCurve\0"
    "loadSettingsFileRequest\0saveSettingsFileRequest\0"
    "sendCommand\0sender\0targetPlugin\0command\0"
    "params\0setKLinCoeffsRequestAccepted\0"
    "setDispCompCoeffsRequestAccepted"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Plugin[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      16,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
      14,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   94,    2, 0x06 /* Public */,
       3,    1,   97,    2, 0x06 /* Public */,
       4,    2,  100,    2, 0x06 /* Public */,
       5,    4,  105,    2, 0x06 /* Public */,
      11,    4,  114,    2, 0x06 /* Public */,
      16,    5,  123,    2, 0x06 /* Public */,
      22,    0,  134,    2, 0x06 /* Public */,
      23,    0,  135,    2, 0x06 /* Public */,
      24,    0,  136,    2, 0x06 /* Public */,
      25,    1,  137,    2, 0x06 /* Public */,
      28,    1,  140,    2, 0x06 /* Public */,
      29,    1,  143,    2, 0x06 /* Public */,
      30,    4,  146,    2, 0x06 /* Public */,
      30,    3,  155,    2, 0x26 /* Public | MethodCloned */,

 // slots: name, argc, parameters, tag, flags
      35,    4,  162,    2, 0x0a /* Public */,
      36,    4,  171,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString,    2,
    QMetaType::Void, QMetaType::QString,    2,
    QMetaType::Void, QMetaType::QString, QMetaType::QVariantMap,    2,    2,
    QMetaType::Void, 0x80000000 | 6, 0x80000000 | 6, 0x80000000 | 6, 0x80000000 | 6,    7,    8,    9,   10,
    QMetaType::Void, 0x80000000 | 6, 0x80000000 | 6, 0x80000000 | 6, 0x80000000 | 6,   12,   13,   14,   15,
    QMetaType::Void, QMetaType::Bool, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double,   17,   18,   19,   20,   21,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 26,   27,
    QMetaType::Void, QMetaType::QString,    2,
    QMetaType::Void, QMetaType::QString,    2,
    QMetaType::Void, QMetaType::QString, QMetaType::QString, QMetaType::QString, QMetaType::QVariantMap,   31,   32,   33,   34,
    QMetaType::Void, QMetaType::QString, QMetaType::QString, QMetaType::QString,   31,   32,   33,

 // slots: parameters
    QMetaType::Void, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double,    7,    8,    9,   10,
    QMetaType::Void, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double,   12,   13,   14,   15,

       0        // eod
};

void Plugin::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<Plugin *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->info((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 1: _t->error((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 2: _t->storeSettings((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QVariantMap(*)>(_a[2]))); break;
        case 3: _t->setKLinCoeffsRequest((*reinterpret_cast< double*(*)>(_a[1])),(*reinterpret_cast< double*(*)>(_a[2])),(*reinterpret_cast< double*(*)>(_a[3])),(*reinterpret_cast< double*(*)>(_a[4]))); break;
        case 4: _t->setDispCompCoeffsRequest((*reinterpret_cast< double*(*)>(_a[1])),(*reinterpret_cast< double*(*)>(_a[2])),(*reinterpret_cast< double*(*)>(_a[3])),(*reinterpret_cast< double*(*)>(_a[4]))); break;
        case 5: _t->setGrayscaleConversionRequest((*reinterpret_cast< bool(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5]))); break;
        case 6: _t->startProcessingRequest(); break;
        case 7: _t->stopProcessingRequest(); break;
        case 8: _t->startRecordingRequest(); break;
        case 9: _t->setCustomResamplingCurveRequest((*reinterpret_cast< QVector<float>(*)>(_a[1]))); break;
        case 10: _t->loadSettingsFileRequest((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 11: _t->saveSettingsFileRequest((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 12: _t->sendCommand((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2])),(*reinterpret_cast< const QString(*)>(_a[3])),(*reinterpret_cast< const QVariantMap(*)>(_a[4]))); break;
        case 13: _t->sendCommand((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2])),(*reinterpret_cast< const QString(*)>(_a[3]))); break;
        case 14: _t->setKLinCoeffsRequestAccepted((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4]))); break;
        case 15: _t->setDispCompCoeffsRequestAccepted((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 9:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QVector<float> >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (Plugin::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::info)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (Plugin::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::error)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (Plugin::*)(QString , QVariantMap );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::storeSettings)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (Plugin::*)(double * , double * , double * , double * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::setKLinCoeffsRequest)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (Plugin::*)(double * , double * , double * , double * );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::setDispCompCoeffsRequest)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (Plugin::*)(bool , double , double , double , double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::setGrayscaleConversionRequest)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (Plugin::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::startProcessingRequest)) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (Plugin::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::stopProcessingRequest)) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (Plugin::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::startRecordingRequest)) {
                *result = 8;
                return;
            }
        }
        {
            using _t = void (Plugin::*)(QVector<float> );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::setCustomResamplingCurveRequest)) {
                *result = 9;
                return;
            }
        }
        {
            using _t = void (Plugin::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::loadSettingsFileRequest)) {
                *result = 10;
                return;
            }
        }
        {
            using _t = void (Plugin::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::saveSettingsFileRequest)) {
                *result = 11;
                return;
            }
        }
        {
            using _t = void (Plugin::*)(const QString & , const QString & , const QString & , const QVariantMap & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Plugin::sendCommand)) {
                *result = 12;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject Plugin::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_Plugin.data,
    qt_meta_data_Plugin,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *Plugin::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Plugin::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_Plugin.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int Plugin::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 16)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 16;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 16)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 16;
    }
    return _id;
}

// SIGNAL 0
void Plugin::info(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void Plugin::error(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void Plugin::storeSettings(QString _t1, QVariantMap _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void Plugin::setKLinCoeffsRequest(double * _t1, double * _t2, double * _t3, double * _t4)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void Plugin::setDispCompCoeffsRequest(double * _t1, double * _t2, double * _t3, double * _t4)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void Plugin::setGrayscaleConversionRequest(bool _t1, double _t2, double _t3, double _t4, double _t5)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t5))) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void Plugin::startProcessingRequest()
{
    QMetaObject::activate(this, &staticMetaObject, 6, nullptr);
}

// SIGNAL 7
void Plugin::stopProcessingRequest()
{
    QMetaObject::activate(this, &staticMetaObject, 7, nullptr);
}

// SIGNAL 8
void Plugin::startRecordingRequest()
{
    QMetaObject::activate(this, &staticMetaObject, 8, nullptr);
}

// SIGNAL 9
void Plugin::setCustomResamplingCurveRequest(QVector<float> _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 9, _a);
}

// SIGNAL 10
void Plugin::loadSettingsFileRequest(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 10, _a);
}

// SIGNAL 11
void Plugin::saveSettingsFileRequest(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 11, _a);
}

// SIGNAL 12
void Plugin::sendCommand(const QString & _t1, const QString & _t2, const QString & _t3, const QVariantMap & _t4)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))) };
    QMetaObject::activate(this, &staticMetaObject, 12, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
