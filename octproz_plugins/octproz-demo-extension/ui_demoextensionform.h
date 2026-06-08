/********************************************************************************
** Form generated from reading UI file 'demoextensionform.ui'
**
** Created by: Qt User Interface Compiler version 5.15.13
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DEMOEXTENSIONFORM_H
#define UI_DEMOEXTENSIONFORM_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DemoExtensionForm
{
public:
    QVBoxLayout *verticalLayout;
    QSpacerItem *verticalSpacer_2;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer_3;
    QLabel *label;
    QSpacerItem *horizontalSpacer_4;
    QHBoxLayout *horizontalLayout_2;
    QSpacerItem *horizontalSpacer;
    QCheckBox *checkBox_like;
    QSpacerItem *horizontalSpacer_2;
    QSpacerItem *verticalSpacer_3;
    QHBoxLayout *horizontalLayout_3;
    QSpacerItem *horizontalSpacer_5;
    QLabel *label_2;
    QDoubleSpinBox *doubleSpinBox_demoParameter;
    QSpacerItem *horizontalSpacer_6;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *DemoExtensionForm)
    {
        if (DemoExtensionForm->objectName().isEmpty())
            DemoExtensionForm->setObjectName(QString::fromUtf8("DemoExtensionForm"));
        DemoExtensionForm->resize(204, 379);
        verticalLayout = new QVBoxLayout(DemoExtensionForm);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalSpacer_2 = new QSpacerItem(20, 146, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer_2);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_3);

        label = new QLabel(DemoExtensionForm);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_4);


        verticalLayout->addLayout(horizontalLayout);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);

        checkBox_like = new QCheckBox(DemoExtensionForm);
        checkBox_like->setObjectName(QString::fromUtf8("checkBox_like"));

        horizontalLayout_2->addWidget(checkBox_like);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_2);


        verticalLayout->addLayout(horizontalLayout_2);

        verticalSpacer_3 = new QSpacerItem(20, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer_3);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_5);

        label_2 = new QLabel(DemoExtensionForm);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_3->addWidget(label_2);

        doubleSpinBox_demoParameter = new QDoubleSpinBox(DemoExtensionForm);
        doubleSpinBox_demoParameter->setObjectName(QString::fromUtf8("doubleSpinBox_demoParameter"));

        horizontalLayout_3->addWidget(doubleSpinBox_demoParameter);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_6);


        verticalLayout->addLayout(horizontalLayout_3);

        verticalSpacer = new QSpacerItem(20, 147, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        retranslateUi(DemoExtensionForm);

        QMetaObject::connectSlotsByName(DemoExtensionForm);
    } // setupUi

    void retranslateUi(QWidget *DemoExtensionForm)
    {
        DemoExtensionForm->setWindowTitle(QCoreApplication::translate("DemoExtensionForm", "Form", nullptr));
        label->setText(QCoreApplication::translate("DemoExtensionForm", "This is a demo extension!", nullptr));
        checkBox_like->setText(QCoreApplication::translate("DemoExtensionForm", "Like", nullptr));
        label_2->setText(QCoreApplication::translate("DemoExtensionForm", "Demo parameter: ", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DemoExtensionForm: public Ui_DemoExtensionForm {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DEMOEXTENSIONFORM_H
