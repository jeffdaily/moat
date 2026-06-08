/********************************************************************************
** Form generated from reading UI file 'virtualoctsystemsettingsdialog.ui'
**
** Created by: Qt User Interface Compiler version 5.15.13
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VIRTUALOCTSYSTEMSETTINGSDIALOG_H
#define UI_VIRTUALOCTSYSTEMSETTINGSDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_VirtualOCTSystemSettingsDialog
{
public:
    QHBoxLayout *horizontalLayout_10;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label;
    QLineEdit *lineEdit;
    QPushButton *pushButton_selectFile;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_6;
    QSpacerItem *spacer_5;
    QSpinBox *spinBox_bitDepth;
    QHBoxLayout *horizontalLayout;
    QLabel *label_2;
    QSpacerItem *spacer;
    QSpinBox *spinBox_width;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_3;
    QSpacerItem *spacer_2;
    QSpinBox *spinBox_height;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_4;
    QSpacerItem *spacer_3;
    QSpinBox *spinBox_depth;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_7;
    QSpacerItem *spacer_6;
    QSpinBox *spinBox_buffersPerVolume;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label_8;
    QSpacerItem *spacer_7;
    QSpinBox *spinBox_buffersFromFile;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label_10;
    QSpacerItem *spacer_8;
    QSpinBox *spinBox_bscanOffset;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_5;
    QSpacerItem *spacer_4;
    QSpinBox *spinBox_waitTime;
    QSpacerItem *verticalSpacer;
    QCheckBox *checkBox_copyFileToRam;
    QCheckBox *checkBox_sync;
    QHBoxLayout *hboxLayout;
    QSpacerItem *spacerItem;
    QPushButton *cancelButton;
    QPushButton *okButton;
    QSpacerItem *horizontalSpacer;
    QLabel *label_9;

    void setupUi(QDialog *VirtualOCTSystemSettingsDialog)
    {
        if (VirtualOCTSystemSettingsDialog->objectName().isEmpty())
            VirtualOCTSystemSettingsDialog->setObjectName(QString::fromUtf8("VirtualOCTSystemSettingsDialog"));
        VirtualOCTSystemSettingsDialog->resize(632, 362);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(VirtualOCTSystemSettingsDialog->sizePolicy().hasHeightForWidth());
        VirtualOCTSystemSettingsDialog->setSizePolicy(sizePolicy);
        horizontalLayout_10 = new QHBoxLayout(VirtualOCTSystemSettingsDialog);
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(3);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label = new QLabel(VirtualOCTSystemSettingsDialog);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_4->addWidget(label);

        lineEdit = new QLineEdit(VirtualOCTSystemSettingsDialog);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));
        lineEdit->setFrame(false);
        lineEdit->setReadOnly(true);

        horizontalLayout_4->addWidget(lineEdit);

        pushButton_selectFile = new QPushButton(VirtualOCTSystemSettingsDialog);
        pushButton_selectFile->setObjectName(QString::fromUtf8("pushButton_selectFile"));

        horizontalLayout_4->addWidget(pushButton_selectFile);


        verticalLayout->addLayout(horizontalLayout_4);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        label_6 = new QLabel(VirtualOCTSystemSettingsDialog);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout_6->addWidget(label_6);

        spacer_5 = new QSpacerItem(38, 13, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(spacer_5);

        spinBox_bitDepth = new QSpinBox(VirtualOCTSystemSettingsDialog);
        spinBox_bitDepth->setObjectName(QString::fromUtf8("spinBox_bitDepth"));
        spinBox_bitDepth->setMinimum(8);
        spinBox_bitDepth->setMaximum(10000000);
        spinBox_bitDepth->setValue(12);

        horizontalLayout_6->addWidget(spinBox_bitDepth);


        verticalLayout->addLayout(horizontalLayout_6);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_2 = new QLabel(VirtualOCTSystemSettingsDialog);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout->addWidget(label_2);

        spacer = new QSpacerItem(38, 13, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(spacer);

        spinBox_width = new QSpinBox(VirtualOCTSystemSettingsDialog);
        spinBox_width->setObjectName(QString::fromUtf8("spinBox_width"));
        spinBox_width->setMinimum(2);
        spinBox_width->setMaximum(10000000);
        spinBox_width->setSingleStep(2);
        spinBox_width->setValue(1024);

        horizontalLayout->addWidget(spinBox_width);


        verticalLayout->addLayout(horizontalLayout);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_3 = new QLabel(VirtualOCTSystemSettingsDialog);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_2->addWidget(label_3);

        spacer_2 = new QSpacerItem(38, 13, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(spacer_2);

        spinBox_height = new QSpinBox(VirtualOCTSystemSettingsDialog);
        spinBox_height->setObjectName(QString::fromUtf8("spinBox_height"));
        spinBox_height->setMinimum(2);
        spinBox_height->setMaximum(10000000);
        spinBox_height->setValue(512);

        horizontalLayout_2->addWidget(spinBox_height);


        verticalLayout->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label_4 = new QLabel(VirtualOCTSystemSettingsDialog);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_3->addWidget(label_4);

        spacer_3 = new QSpacerItem(38, 13, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(spacer_3);

        spinBox_depth = new QSpinBox(VirtualOCTSystemSettingsDialog);
        spinBox_depth->setObjectName(QString::fromUtf8("spinBox_depth"));
        spinBox_depth->setMinimum(1);
        spinBox_depth->setMaximum(10000000);
        spinBox_depth->setValue(64);

        horizontalLayout_3->addWidget(spinBox_depth);


        verticalLayout->addLayout(horizontalLayout_3);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        label_7 = new QLabel(VirtualOCTSystemSettingsDialog);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        horizontalLayout_7->addWidget(label_7);

        spacer_6 = new QSpacerItem(38, 13, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_7->addItem(spacer_6);

        spinBox_buffersPerVolume = new QSpinBox(VirtualOCTSystemSettingsDialog);
        spinBox_buffersPerVolume->setObjectName(QString::fromUtf8("spinBox_buffersPerVolume"));
        spinBox_buffersPerVolume->setMinimum(1);
        spinBox_buffersPerVolume->setMaximum(10000000);
        spinBox_buffersPerVolume->setValue(1);

        horizontalLayout_7->addWidget(spinBox_buffersPerVolume);


        verticalLayout->addLayout(horizontalLayout_7);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        label_8 = new QLabel(VirtualOCTSystemSettingsDialog);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        horizontalLayout_8->addWidget(label_8);

        spacer_7 = new QSpacerItem(38, 13, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_8->addItem(spacer_7);

        spinBox_buffersFromFile = new QSpinBox(VirtualOCTSystemSettingsDialog);
        spinBox_buffersFromFile->setObjectName(QString::fromUtf8("spinBox_buffersFromFile"));
        spinBox_buffersFromFile->setMinimum(1);
        spinBox_buffersFromFile->setMaximum(10000000);
        spinBox_buffersFromFile->setValue(1);

        horizontalLayout_8->addWidget(spinBox_buffersFromFile);


        verticalLayout->addLayout(horizontalLayout_8);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        label_10 = new QLabel(VirtualOCTSystemSettingsDialog);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        horizontalLayout_9->addWidget(label_10);

        spacer_8 = new QSpacerItem(38, 13, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_9->addItem(spacer_8);

        spinBox_bscanOffset = new QSpinBox(VirtualOCTSystemSettingsDialog);
        spinBox_bscanOffset->setObjectName(QString::fromUtf8("spinBox_bscanOffset"));
        spinBox_bscanOffset->setMinimum(0);
        spinBox_bscanOffset->setMaximum(10000000);
        spinBox_bscanOffset->setValue(0);

        horizontalLayout_9->addWidget(spinBox_bscanOffset);


        verticalLayout->addLayout(horizontalLayout_9);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_5 = new QLabel(VirtualOCTSystemSettingsDialog);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        horizontalLayout_5->addWidget(label_5);

        spacer_4 = new QSpacerItem(38, 13, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(spacer_4);

        spinBox_waitTime = new QSpinBox(VirtualOCTSystemSettingsDialog);
        spinBox_waitTime->setObjectName(QString::fromUtf8("spinBox_waitTime"));
        spinBox_waitTime->setMaximum(10000000);
        spinBox_waitTime->setValue(10);

        horizontalLayout_5->addWidget(spinBox_waitTime);


        verticalLayout->addLayout(horizontalLayout_5);

        verticalSpacer = new QSpacerItem(20, 13, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);

        checkBox_copyFileToRam = new QCheckBox(VirtualOCTSystemSettingsDialog);
        checkBox_copyFileToRam->setObjectName(QString::fromUtf8("checkBox_copyFileToRam"));

        verticalLayout->addWidget(checkBox_copyFileToRam);

        checkBox_sync = new QCheckBox(VirtualOCTSystemSettingsDialog);
        checkBox_sync->setObjectName(QString::fromUtf8("checkBox_sync"));
        checkBox_sync->setChecked(true);

        verticalLayout->addWidget(checkBox_sync);

        hboxLayout = new QHBoxLayout();
        hboxLayout->setSpacing(6);
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        hboxLayout->setContentsMargins(0, 0, 0, 0);
        spacerItem = new QSpacerItem(131, 31, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout->addItem(spacerItem);

        cancelButton = new QPushButton(VirtualOCTSystemSettingsDialog);
        cancelButton->setObjectName(QString::fromUtf8("cancelButton"));

        hboxLayout->addWidget(cancelButton);

        okButton = new QPushButton(VirtualOCTSystemSettingsDialog);
        okButton->setObjectName(QString::fromUtf8("okButton"));

        hboxLayout->addWidget(okButton);


        verticalLayout->addLayout(hboxLayout);


        horizontalLayout_10->addLayout(verticalLayout);

        horizontalSpacer = new QSpacerItem(20, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

        horizontalLayout_10->addItem(horizontalSpacer);

        label_9 = new QLabel(VirtualOCTSystemSettingsDialog);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        label_9->setPixmap(QPixmap(QString::fromUtf8(":/images/volumeparams.png")));

        horizontalLayout_10->addWidget(label_9);


        retranslateUi(VirtualOCTSystemSettingsDialog);
        QObject::connect(okButton, SIGNAL(clicked()), VirtualOCTSystemSettingsDialog, SLOT(accept()));
        QObject::connect(cancelButton, SIGNAL(clicked()), VirtualOCTSystemSettingsDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(VirtualOCTSystemSettingsDialog);
    } // setupUi

    void retranslateUi(QDialog *VirtualOCTSystemSettingsDialog)
    {
        VirtualOCTSystemSettingsDialog->setWindowTitle(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Dialog", nullptr));
        label->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "File:", nullptr));
        lineEdit->setText(QString());
        pushButton_selectFile->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Select file", nullptr));
        label_6->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Bit depth [bits]:", nullptr));
#if QT_CONFIG(tooltip)
        spinBox_bitDepth->setToolTip(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "<html><head/><body><p>Bit depth of each sample.</p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        spinBox_bitDepth->setWhatsThis(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Bit depth of each sample. The exact value depends on the dimensions of the chosen raw data. Common values are 8 bit, 12 bit and 16 bit.", nullptr));
#endif // QT_CONFIG(whatsthis)
        label_2->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Samples per raw A-scan:", nullptr));
#if QT_CONFIG(tooltip)
        spinBox_width->setToolTip(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "<html><head/><body><p>Samples per raw a-scan. This needs to be an even number. </p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        spinBox_width->setWhatsThis(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Samples per raw a-scan. This needs to be an even number. The exact value depends on the dimensions of the chosen raw data. Typical values are multiple of 128 such as 1024.", nullptr));
#endif // QT_CONFIG(whatsthis)
        label_3->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "A-scans per B-scan:", nullptr));
#if QT_CONFIG(tooltip)
        spinBox_height->setToolTip(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "<html><head/><body><p>A-scans per b-scan</p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        spinBox_height->setWhatsThis(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "A-scans per b-scan. The exact value depends on the dimensions of the chosen raw data. Typical values are 256, 320, 512 and 1024.", nullptr));
#endif // QT_CONFIG(whatsthis)
        label_4->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "B-scans per buffer:", nullptr));
#if QT_CONFIG(tooltip)
        spinBox_depth->setToolTip(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "<html><head/><body><p>B-scans per buffer</p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        spinBox_depth->setWhatsThis(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", " For processing buffer by buffer is copied to GPU memory and all a-scans of each buffer are processed in parallel. The size of the buffer affects the processing speed. The optimal value depends on the GPU used.", nullptr));
#endif // QT_CONFIG(whatsthis)
        label_7->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Buffers per volume:", nullptr));
#if QT_CONFIG(tooltip)
        spinBox_buffersPerVolume->setToolTip(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "<html><head/><body><p>Buffers per volume</p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        spinBox_buffersPerVolume->setWhatsThis(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Buffers per volume. A volume can consist of one or more buffers. For processing buffer by buffer is copied to GPU memory and all a-scans of each buffer are processed in parallel.", nullptr));
#endif // QT_CONFIG(whatsthis)
        label_8->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Buffers to read from file:", nullptr));
#if QT_CONFIG(tooltip)
        spinBox_buffersFromFile->setToolTip(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "<html><head/><body><p>Number of buffers that should be read from file.</p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        spinBox_buffersFromFile->setWhatsThis(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Number of buffers that should be read from file. Typically this value is the same as \"buffers per volume\" or an integer multiple of it.", nullptr));
#endif // QT_CONFIG(whatsthis)
        label_10->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Offset (B-scans to skip):", nullptr));
#if QT_CONFIG(tooltip)
        spinBox_bscanOffset->setToolTip(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "<html><head/><body><p>Number of buffers that should be read from file.</p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        spinBox_bscanOffset->setWhatsThis(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Number of buffers that should be read from file. Typically this value is the same as \"buffers per volume\" or an integer multiple of it.", nullptr));
#endif // QT_CONFIG(whatsthis)
        label_5->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Wait after file read [us]:", nullptr));
#if QT_CONFIG(tooltip)
        spinBox_waitTime->setToolTip(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "<html><head/><body><p>Wait time after reading a single buffer.</p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        spinBox_waitTime->setWhatsThis(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Artificially generated waiting time after reading a single buffer to simulate slow acquisition systems.", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(tooltip)
        checkBox_copyFileToRam->setToolTip(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "<html><head/><body><p>Copy entire raw file to RAM to speed up data transfer to OCTproZ. For very large files this should be disabled. This option only has an effect if buffers to read from file is greater than two.  </p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        checkBox_copyFileToRam->setWhatsThis(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Copy entire raw file to RAM to speed up data transfer to OCTproZ. For very large files this should be disabled. This option only has an effect if buffers to read from file is greater than two. ", nullptr));
#endif // QT_CONFIG(whatsthis)
        checkBox_copyFileToRam->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Copy file to RAM", nullptr));
#if QT_CONFIG(tooltip)
        checkBox_sync->setToolTip(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "<html><head/><body><p>When synchronization is enabled, it is guaranteed that no buffer will be lost. However, it introduces a small idle time between each buffer. When measuring the performance of the processing pipeline, synchronization should be disabled.</p></body></html>", nullptr));
#endif // QT_CONFIG(tooltip)
        checkBox_sync->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Synchronize with processing", nullptr));
        cancelButton->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "Cancel", nullptr));
        okButton->setText(QCoreApplication::translate("VirtualOCTSystemSettingsDialog", "OK", nullptr));
        label_9->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class VirtualOCTSystemSettingsDialog: public Ui_VirtualOCTSystemSettingsDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VIRTUALOCTSYSTEMSETTINGSDIALOG_H
