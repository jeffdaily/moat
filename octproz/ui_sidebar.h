/********************************************************************************
** Form generated from reading UI file 'sidebar.ui'
**
** Created by: Qt User Interface Compiler version 5.15.13
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SIDEBAR_H
#define UI_SIDEBAR_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "minicurveplot.h"

QT_BEGIN_NAMESPACE

class Ui_Sidebar
{
public:
    QWidget *dockWidgetContents;
    QHBoxLayout *horizontalLayout_27;
    QWidget *widget_sidebarContent;
    QVBoxLayout *verticalLayout_18;
    QHBoxLayout *horizontalLayout;
    QToolButton *toolButton_start;
    QToolButton *toolButton_stop;
    QToolButton *toolButton_rec;
    QSpacerItem *horizontalSpacer;
    QToolButton *toolButton_system;
    QToolButton *toolButton_settings;
    QTabWidget *tabWidget;
    QWidget *tab;
    QVBoxLayout *verticalLayout_20;
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *verticalLayout_10;
    QGroupBox *groupBox_rawSignalCorrection;
    QVBoxLayout *verticalLayout_9;
    QCheckBox *checkBox_bitshift;
    QGroupBox *groupBox_5;
    QVBoxLayout *verticalLayout_19;
    QCheckBox *checkBox_sinusoidalScanCorrection;
    QCheckBox *checkBox_bscanFlip;
    QGroupBox *groupBox_backgroundremoval;
    QHBoxLayout *horizontalLayout_25;
    QLabel *label_3;
    QSpinBox *spinBox_rollingAverageWindowSize;
    QSpacerItem *horizontalSpacer_11;
    QGroupBox *groupBox_resampling;
    QVBoxLayout *verticalLayout_7;
    QHBoxLayout *horizontalLayout_31;
    QLabel *label_31;
    QComboBox *comboBox_interpolation;
    QSpacerItem *horizontalSpacer_10;
    QHBoxLayout *horizontalLayout_34;
    QVBoxLayout *verticalLayout_11;
    QHBoxLayout *horizontalLayout_14;
    QLabel *label_12;
    QDoubleSpinBox *doubleSpinBox_c0;
    QHBoxLayout *horizontalLayout_15;
    QLabel *label_13;
    QDoubleSpinBox *doubleSpinBox_c1;
    QHBoxLayout *horizontalLayout_16;
    QLabel *label_14;
    QDoubleSpinBox *doubleSpinBox_c2;
    QHBoxLayout *horizontalLayout_17;
    QLabel *label_15;
    QDoubleSpinBox *doubleSpinBox_c3;
    MiniCurvePlot *widget_resampleCurvePlot;
    QGroupBox *groupBox_dispersionCompensation;
    QHBoxLayout *horizontalLayout_32;
    QVBoxLayout *verticalLayout_12;
    QHBoxLayout *horizontalLayout_18;
    QLabel *label_16;
    QDoubleSpinBox *doubleSpinBox_d0;
    QHBoxLayout *horizontalLayout_19;
    QLabel *label_17;
    QDoubleSpinBox *doubleSpinBox_d1;
    QHBoxLayout *horizontalLayout_20;
    QLabel *label_18;
    QDoubleSpinBox *doubleSpinBox_d2;
    QHBoxLayout *horizontalLayout_21;
    QLabel *label_19;
    QDoubleSpinBox *doubleSpinBox_d3;
    MiniCurvePlot *widget_dispersionCurvePlot;
    QGroupBox *groupBox_windowing;
    QHBoxLayout *horizontalLayout_33;
    QVBoxLayout *verticalLayout_13;
    QHBoxLayout *horizontalLayout_24;
    QLabel *label_22;
    QComboBox *comboBox_windowType;
    QHBoxLayout *horizontalLayout_23;
    QLabel *label_23;
    QDoubleSpinBox *doubleSpinBox_windowFillFactor;
    QHBoxLayout *horizontalLayout_22;
    QLabel *label_24;
    QDoubleSpinBox *doubleSpinBox_windowCenterPosition;
    MiniCurvePlot *widget_windowCurvePlot;
    QGroupBox *groupBox_fixedPatternNoiseRemoval;
    QVBoxLayout *verticalLayout_17;
    QHBoxLayout *horizontalLayout_29;
    QLabel *label_28;
    QSpinBox *spinBox_bscansFixedNoise;
    QLabel *label_29;
    QLabel *label_27;
    QHBoxLayout *horizontalLayout_28;
    QRadioButton *radioButton_once;
    QSpacerItem *horizontalSpacer_9;
    QPushButton *pushButton_redetermine;
    QRadioButton *radioButton_continuously;
    QGroupBox *groupBox_3;
    QVBoxLayout *verticalLayout_6;
    QCheckBox *checkBox_logScaling;
    QHBoxLayout *horizontalLayout_6;
    QLabel *label_4;
    QDoubleSpinBox *doubleSpinBox_signalMax;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_5;
    QDoubleSpinBox *doubleSpinBox_signalMin;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label_6;
    QDoubleSpinBox *doubleSpinBox_signalMultiplicator;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label_7;
    QDoubleSpinBox *doubleSpinBox_signalAddend;
    QGroupBox *groupBox_postProcessBackgroundRemoval;
    QHBoxLayout *horizontalLayout_38;
    QVBoxLayout *verticalLayout_21;
    QHBoxLayout *horizontalLayout_36;
    QPushButton *pushButton_postProcRec;
    QPushButton *pushButton_postProcSave;
    QPushButton *pushButton_postProcLoad;
    QHBoxLayout *horizontalLayout_35;
    QLabel *label_8;
    QSpacerItem *horizontalSpacer_13;
    QDoubleSpinBox *doubleSpinBox_postProcessBackgroundWeight;
    QHBoxLayout *horizontalLayout_37;
    QLabel *label_9;
    QSpacerItem *horizontalSpacer_12;
    QDoubleSpinBox *doubleSpinBox_postProcessBackgroundOffset;
    MiniCurvePlot *widget_postProcessBackgroundPlot;
    QGroupBox *groupBox_streaming;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_26;
    QLabel *label_26;
    QSpinBox *spinBox_streamingBuffersToSkip;
    QSpacerItem *verticalSpacer_3;
    QGroupBox *groupBox_info;
    QVBoxLayout *verticalLayout_8;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_name_volumesPerSecond;
    QSpacerItem *horizontalSpacer_3;
    QLabel *label_volumesPerSecond;
    QHBoxLayout *horizontalLayout_30;
    QLabel *label_name_buffersPerSecond;
    QSpacerItem *horizontalSpacer_8;
    QLabel *label_buffersPerSecond;
    QHBoxLayout *horizontalLayout_12;
    QLabel *label_name_bscansPerSecond;
    QSpacerItem *horizontalSpacer_6;
    QLabel *label_bscansPerSecond;
    QHBoxLayout *horizontalLayout_13;
    QLabel *label_name_ascansPerSecond;
    QSpacerItem *horizontalSpacer_7;
    QLabel *label_ascansPerSecond;
    QHBoxLayout *horizontalLayout_10;
    QLabel *label_name_bufferSize;
    QSpacerItem *horizontalSpacer_4;
    QLabel *label_bufferSize;
    QHBoxLayout *horizontalLayout_11;
    QLabel *label_name_dataThroughput;
    QSpacerItem *horizontalSpacer_5;
    QLabel *label_dataThroughput;
    QWidget *tab_3;
    QVBoxLayout *verticalLayout_16;
    QScrollArea *scrollArea_2;
    QWidget *scrollAreaWidgetContents_2;
    QVBoxLayout *verticalLayout_22;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label;
    QSpacerItem *horizontalSpacer_2;
    QHBoxLayout *horizontalLayout_2;
    QLineEdit *lineEdit_saveFolder;
    QToolButton *toolButton_recPath;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout_3;
    QCheckBox *checkBox_recordScreenshots;
    QCheckBox *checkBox_recordRawBuffers;
    QCheckBox *checkBox_recordProcessedBuffers;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_4;
    QCheckBox *checkBox_startWithFirstBuffer;
    QCheckBox *checkBox_stopAfterRec;
    QCheckBox *checkBox_meta;
    QCheckBox *checkBox_32bitfloat;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_2;
    QSpinBox *spinBox_volumes;
    QGroupBox *groupBox_6;
    QVBoxLayout *verticalLayout_15;
    QVBoxLayout *verticalLayout_14;
    QLabel *label_20;
    QLineEdit *lineEdit_recName;
    QVBoxLayout *verticalLayout_5;
    QLabel *label_21;
    QPlainTextEdit *plainTextEdit_description;
    QPushButton *pushButton_showSidebar;

    void setupUi(QDockWidget *Sidebar)
    {
        if (Sidebar->objectName().isEmpty())
            Sidebar->setObjectName(QString::fromUtf8("Sidebar"));
        Sidebar->setEnabled(true);
        Sidebar->resize(340, 593);
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::MinimumExpanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(Sidebar->sizePolicy().hasHeightForWidth());
        Sidebar->setSizePolicy(sizePolicy);
        Sidebar->setMinimumSize(QSize(340, 317));
        Sidebar->setMaximumSize(QSize(340, 524287));
        Sidebar->setMouseTracking(false);
        Sidebar->setFeatures(QDockWidget::NoDockWidgetFeatures);
        Sidebar->setAllowedAreas(Qt::NoDockWidgetArea);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        QSizePolicy sizePolicy1(QSizePolicy::Maximum, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(dockWidgetContents->sizePolicy().hasHeightForWidth());
        dockWidgetContents->setSizePolicy(sizePolicy1);
        dockWidgetContents->setMinimumSize(QSize(340, 0));
        horizontalLayout_27 = new QHBoxLayout(dockWidgetContents);
        horizontalLayout_27->setSpacing(0);
        horizontalLayout_27->setObjectName(QString::fromUtf8("horizontalLayout_27"));
        horizontalLayout_27->setContentsMargins(0, 0, 0, 1);
        widget_sidebarContent = new QWidget(dockWidgetContents);
        widget_sidebarContent->setObjectName(QString::fromUtf8("widget_sidebarContent"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy2.setHorizontalStretch(2);
        sizePolicy2.setVerticalStretch(2);
        sizePolicy2.setHeightForWidth(widget_sidebarContent->sizePolicy().hasHeightForWidth());
        widget_sidebarContent->setSizePolicy(sizePolicy2);
        verticalLayout_18 = new QVBoxLayout(widget_sidebarContent);
        verticalLayout_18->setSpacing(0);
        verticalLayout_18->setObjectName(QString::fromUtf8("verticalLayout_18"));
        verticalLayout_18->setContentsMargins(0, 0, 0, 0);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 6, -1, 6);
        toolButton_start = new QToolButton(widget_sidebarContent);
        toolButton_start->setObjectName(QString::fromUtf8("toolButton_start"));
        QIcon icon;
        icon.addFile(QString::fromUtf8("../icons/play_white.png"), QSize(), QIcon::Normal, QIcon::Off);
        toolButton_start->setIcon(icon);
        toolButton_start->setIconSize(QSize(32, 32));
        toolButton_start->setCheckable(true);
        toolButton_start->setChecked(false);
        toolButton_start->setAutoRepeat(false);
        toolButton_start->setPopupMode(QToolButton::DelayedPopup);
        toolButton_start->setToolButtonStyle(Qt::ToolButtonIconOnly);
        toolButton_start->setAutoRaise(true);
        toolButton_start->setArrowType(Qt::NoArrow);

        horizontalLayout->addWidget(toolButton_start);

        toolButton_stop = new QToolButton(widget_sidebarContent);
        toolButton_stop->setObjectName(QString::fromUtf8("toolButton_stop"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8("../icons/stop_white.png"), QSize(), QIcon::Normal, QIcon::Off);
        toolButton_stop->setIcon(icon1);
        toolButton_stop->setIconSize(QSize(32, 32));
        toolButton_stop->setCheckable(true);
        toolButton_stop->setChecked(false);
        toolButton_stop->setAutoRepeat(false);
        toolButton_stop->setPopupMode(QToolButton::DelayedPopup);
        toolButton_stop->setToolButtonStyle(Qt::ToolButtonIconOnly);
        toolButton_stop->setAutoRaise(true);
        toolButton_stop->setArrowType(Qt::NoArrow);

        horizontalLayout->addWidget(toolButton_stop);

        toolButton_rec = new QToolButton(widget_sidebarContent);
        toolButton_rec->setObjectName(QString::fromUtf8("toolButton_rec"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8("../icons/rec_white.png"), QSize(), QIcon::Normal, QIcon::Off);
        toolButton_rec->setIcon(icon2);
        toolButton_rec->setIconSize(QSize(32, 32));
        toolButton_rec->setCheckable(true);
        toolButton_rec->setChecked(false);
        toolButton_rec->setAutoRepeat(false);
        toolButton_rec->setPopupMode(QToolButton::DelayedPopup);
        toolButton_rec->setToolButtonStyle(Qt::ToolButtonIconOnly);
        toolButton_rec->setAutoRaise(true);
        toolButton_rec->setArrowType(Qt::NoArrow);

        horizontalLayout->addWidget(toolButton_rec);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        toolButton_system = new QToolButton(widget_sidebarContent);
        toolButton_system->setObjectName(QString::fromUtf8("toolButton_system"));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8("../icons/octproz_connect_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
        toolButton_system->setIcon(icon3);
        toolButton_system->setIconSize(QSize(32, 32));
        toolButton_system->setToolButtonStyle(Qt::ToolButtonIconOnly);
        toolButton_system->setAutoRaise(true);

        horizontalLayout->addWidget(toolButton_system);

        toolButton_settings = new QToolButton(widget_sidebarContent);
        toolButton_settings->setObjectName(QString::fromUtf8("toolButton_settings"));
        toolButton_settings->setEnabled(true);
        QIcon icon4;
        icon4.addFile(QString::fromUtf8("../icons/octproz_settings_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
        toolButton_settings->setIcon(icon4);
        toolButton_settings->setIconSize(QSize(32, 32));
        toolButton_settings->setCheckable(false);
        toolButton_settings->setChecked(false);
        toolButton_settings->setAutoRepeat(false);
        toolButton_settings->setPopupMode(QToolButton::DelayedPopup);
        toolButton_settings->setToolButtonStyle(Qt::ToolButtonIconOnly);
        toolButton_settings->setAutoRaise(true);
        toolButton_settings->setArrowType(Qt::NoArrow);

        horizontalLayout->addWidget(toolButton_settings);


        verticalLayout_18->addLayout(horizontalLayout);

        tabWidget = new QTabWidget(widget_sidebarContent);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        QSizePolicy sizePolicy3(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(tabWidget->sizePolicy().hasHeightForWidth());
        tabWidget->setSizePolicy(sizePolicy3);
        tabWidget->setElideMode(Qt::ElideNone);
        tabWidget->setDocumentMode(true);
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        sizePolicy3.setHeightForWidth(tab->sizePolicy().hasHeightForWidth());
        tab->setSizePolicy(sizePolicy3);
        verticalLayout_20 = new QVBoxLayout(tab);
        verticalLayout_20->setSpacing(6);
        verticalLayout_20->setObjectName(QString::fromUtf8("verticalLayout_20"));
        verticalLayout_20->setContentsMargins(3, -1, 3, -1);
        scrollArea = new QScrollArea(tab);
        scrollArea->setObjectName(QString::fromUtf8("scrollArea"));
        sizePolicy3.setHeightForWidth(scrollArea->sizePolicy().hasHeightForWidth());
        scrollArea->setSizePolicy(sizePolicy3);
        scrollArea->setFrameShape(QFrame::NoFrame);
        scrollArea->setFrameShadow(QFrame::Plain);
        scrollArea->setLineWidth(0);
        scrollArea->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 310, 1039));
        QSizePolicy sizePolicy4(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(scrollAreaWidgetContents->sizePolicy().hasHeightForWidth());
        scrollAreaWidgetContents->setSizePolicy(sizePolicy4);
        verticalLayout_10 = new QVBoxLayout(scrollAreaWidgetContents);
        verticalLayout_10->setSpacing(6);
        verticalLayout_10->setObjectName(QString::fromUtf8("verticalLayout_10"));
        verticalLayout_10->setContentsMargins(0, 0, 3, 0);
        groupBox_rawSignalCorrection = new QGroupBox(scrollAreaWidgetContents);
        groupBox_rawSignalCorrection->setObjectName(QString::fromUtf8("groupBox_rawSignalCorrection"));
        verticalLayout_9 = new QVBoxLayout(groupBox_rawSignalCorrection);
        verticalLayout_9->setSpacing(0);
        verticalLayout_9->setObjectName(QString::fromUtf8("verticalLayout_9"));
        verticalLayout_9->setContentsMargins(3, 3, 3, 3);
        checkBox_bitshift = new QCheckBox(groupBox_rawSignalCorrection);
        checkBox_bitshift->setObjectName(QString::fromUtf8("checkBox_bitshift"));

        verticalLayout_9->addWidget(checkBox_bitshift);


        verticalLayout_10->addWidget(groupBox_rawSignalCorrection);

        groupBox_5 = new QGroupBox(scrollAreaWidgetContents);
        groupBox_5->setObjectName(QString::fromUtf8("groupBox_5"));
        verticalLayout_19 = new QVBoxLayout(groupBox_5);
        verticalLayout_19->setSpacing(0);
        verticalLayout_19->setObjectName(QString::fromUtf8("verticalLayout_19"));
        verticalLayout_19->setContentsMargins(3, 3, 3, 3);
        checkBox_sinusoidalScanCorrection = new QCheckBox(groupBox_5);
        checkBox_sinusoidalScanCorrection->setObjectName(QString::fromUtf8("checkBox_sinusoidalScanCorrection"));
        QSizePolicy sizePolicy5(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(checkBox_sinusoidalScanCorrection->sizePolicy().hasHeightForWidth());
        checkBox_sinusoidalScanCorrection->setSizePolicy(sizePolicy5);

        verticalLayout_19->addWidget(checkBox_sinusoidalScanCorrection);

        checkBox_bscanFlip = new QCheckBox(groupBox_5);
        checkBox_bscanFlip->setObjectName(QString::fromUtf8("checkBox_bscanFlip"));

        verticalLayout_19->addWidget(checkBox_bscanFlip);


        verticalLayout_10->addWidget(groupBox_5);

        groupBox_backgroundremoval = new QGroupBox(scrollAreaWidgetContents);
        groupBox_backgroundremoval->setObjectName(QString::fromUtf8("groupBox_backgroundremoval"));
        groupBox_backgroundremoval->setCheckable(true);
        horizontalLayout_25 = new QHBoxLayout(groupBox_backgroundremoval);
        horizontalLayout_25->setSpacing(3);
        horizontalLayout_25->setObjectName(QString::fromUtf8("horizontalLayout_25"));
        horizontalLayout_25->setContentsMargins(3, 3, 3, 3);
        label_3 = new QLabel(groupBox_backgroundremoval);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout_25->addWidget(label_3);

        spinBox_rollingAverageWindowSize = new QSpinBox(groupBox_backgroundremoval);
        spinBox_rollingAverageWindowSize->setObjectName(QString::fromUtf8("spinBox_rollingAverageWindowSize"));
        sizePolicy5.setHeightForWidth(spinBox_rollingAverageWindowSize->sizePolicy().hasHeightForWidth());
        spinBox_rollingAverageWindowSize->setSizePolicy(sizePolicy5);
        spinBox_rollingAverageWindowSize->setMinimum(1);
        spinBox_rollingAverageWindowSize->setMaximum(1024000);

        horizontalLayout_25->addWidget(spinBox_rollingAverageWindowSize);

        horizontalSpacer_11 = new QSpacerItem(142, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_25->addItem(horizontalSpacer_11);


        verticalLayout_10->addWidget(groupBox_backgroundremoval);

        groupBox_resampling = new QGroupBox(scrollAreaWidgetContents);
        groupBox_resampling->setObjectName(QString::fromUtf8("groupBox_resampling"));
        groupBox_resampling->setContextMenuPolicy(Qt::ActionsContextMenu);
        groupBox_resampling->setFlat(false);
        groupBox_resampling->setCheckable(true);
        verticalLayout_7 = new QVBoxLayout(groupBox_resampling);
        verticalLayout_7->setSpacing(3);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        verticalLayout_7->setContentsMargins(3, 3, 3, 3);
        horizontalLayout_31 = new QHBoxLayout();
        horizontalLayout_31->setObjectName(QString::fromUtf8("horizontalLayout_31"));
        label_31 = new QLabel(groupBox_resampling);
        label_31->setObjectName(QString::fromUtf8("label_31"));

        horizontalLayout_31->addWidget(label_31);

        comboBox_interpolation = new QComboBox(groupBox_resampling);
        comboBox_interpolation->setObjectName(QString::fromUtf8("comboBox_interpolation"));
        sizePolicy5.setHeightForWidth(comboBox_interpolation->sizePolicy().hasHeightForWidth());
        comboBox_interpolation->setSizePolicy(sizePolicy5);

        horizontalLayout_31->addWidget(comboBox_interpolation);

        horizontalSpacer_10 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_31->addItem(horizontalSpacer_10);


        verticalLayout_7->addLayout(horizontalLayout_31);

        horizontalLayout_34 = new QHBoxLayout();
        horizontalLayout_34->setSpacing(3);
        horizontalLayout_34->setObjectName(QString::fromUtf8("horizontalLayout_34"));
        verticalLayout_11 = new QVBoxLayout();
        verticalLayout_11->setSpacing(0);
        verticalLayout_11->setObjectName(QString::fromUtf8("verticalLayout_11"));
        horizontalLayout_14 = new QHBoxLayout();
        horizontalLayout_14->setObjectName(QString::fromUtf8("horizontalLayout_14"));
        label_12 = new QLabel(groupBox_resampling);
        label_12->setObjectName(QString::fromUtf8("label_12"));
        QSizePolicy sizePolicy6(QSizePolicy::Maximum, QSizePolicy::Preferred);
        sizePolicy6.setHorizontalStretch(0);
        sizePolicy6.setVerticalStretch(0);
        sizePolicy6.setHeightForWidth(label_12->sizePolicy().hasHeightForWidth());
        label_12->setSizePolicy(sizePolicy6);

        horizontalLayout_14->addWidget(label_12);

        doubleSpinBox_c0 = new QDoubleSpinBox(groupBox_resampling);
        doubleSpinBox_c0->setObjectName(QString::fromUtf8("doubleSpinBox_c0"));
        sizePolicy5.setHeightForWidth(doubleSpinBox_c0->sizePolicy().hasHeightForWidth());
        doubleSpinBox_c0->setSizePolicy(sizePolicy5);
        doubleSpinBox_c0->setMinimumSize(QSize(100, 0));
        doubleSpinBox_c0->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_c0->setKeyboardTracking(true);
        doubleSpinBox_c0->setProperty("showGroupSeparator", QVariant(false));
        doubleSpinBox_c0->setDecimals(6);
        doubleSpinBox_c0->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_c0->setMaximum(10000000000000000.000000000000000);
        doubleSpinBox_c0->setSingleStep(10.000000000000000);

        horizontalLayout_14->addWidget(doubleSpinBox_c0);


        verticalLayout_11->addLayout(horizontalLayout_14);

        horizontalLayout_15 = new QHBoxLayout();
        horizontalLayout_15->setObjectName(QString::fromUtf8("horizontalLayout_15"));
        label_13 = new QLabel(groupBox_resampling);
        label_13->setObjectName(QString::fromUtf8("label_13"));
        sizePolicy6.setHeightForWidth(label_13->sizePolicy().hasHeightForWidth());
        label_13->setSizePolicy(sizePolicy6);

        horizontalLayout_15->addWidget(label_13);

        doubleSpinBox_c1 = new QDoubleSpinBox(groupBox_resampling);
        doubleSpinBox_c1->setObjectName(QString::fromUtf8("doubleSpinBox_c1"));
        sizePolicy5.setHeightForWidth(doubleSpinBox_c1->sizePolicy().hasHeightForWidth());
        doubleSpinBox_c1->setSizePolicy(sizePolicy5);
        doubleSpinBox_c1->setMinimumSize(QSize(40, 0));
        doubleSpinBox_c1->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_c1->setDecimals(6);
        doubleSpinBox_c1->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_c1->setMaximum(10000000000000000.000000000000000);
        doubleSpinBox_c1->setSingleStep(1.000000000000000);

        horizontalLayout_15->addWidget(doubleSpinBox_c1);


        verticalLayout_11->addLayout(horizontalLayout_15);

        horizontalLayout_16 = new QHBoxLayout();
        horizontalLayout_16->setObjectName(QString::fromUtf8("horizontalLayout_16"));
        label_14 = new QLabel(groupBox_resampling);
        label_14->setObjectName(QString::fromUtf8("label_14"));
        sizePolicy6.setHeightForWidth(label_14->sizePolicy().hasHeightForWidth());
        label_14->setSizePolicy(sizePolicy6);

        horizontalLayout_16->addWidget(label_14);

        doubleSpinBox_c2 = new QDoubleSpinBox(groupBox_resampling);
        doubleSpinBox_c2->setObjectName(QString::fromUtf8("doubleSpinBox_c2"));
        sizePolicy5.setHeightForWidth(doubleSpinBox_c2->sizePolicy().hasHeightForWidth());
        doubleSpinBox_c2->setSizePolicy(sizePolicy5);
        doubleSpinBox_c2->setMinimumSize(QSize(40, 0));
        doubleSpinBox_c2->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_c2->setDecimals(6);
        doubleSpinBox_c2->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_c2->setMaximum(10000000000000000.000000000000000);
        doubleSpinBox_c2->setSingleStep(5.000000000000000);
        doubleSpinBox_c2->setValue(0.000000000000000);

        horizontalLayout_16->addWidget(doubleSpinBox_c2);


        verticalLayout_11->addLayout(horizontalLayout_16);

        horizontalLayout_17 = new QHBoxLayout();
        horizontalLayout_17->setObjectName(QString::fromUtf8("horizontalLayout_17"));
        label_15 = new QLabel(groupBox_resampling);
        label_15->setObjectName(QString::fromUtf8("label_15"));
        sizePolicy6.setHeightForWidth(label_15->sizePolicy().hasHeightForWidth());
        label_15->setSizePolicy(sizePolicy6);

        horizontalLayout_17->addWidget(label_15);

        doubleSpinBox_c3 = new QDoubleSpinBox(groupBox_resampling);
        doubleSpinBox_c3->setObjectName(QString::fromUtf8("doubleSpinBox_c3"));
        sizePolicy5.setHeightForWidth(doubleSpinBox_c3->sizePolicy().hasHeightForWidth());
        doubleSpinBox_c3->setSizePolicy(sizePolicy5);
        doubleSpinBox_c3->setMinimumSize(QSize(40, 0));
        doubleSpinBox_c3->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_c3->setDecimals(6);
        doubleSpinBox_c3->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_c3->setMaximum(10000000000000000.000000000000000);
        doubleSpinBox_c3->setSingleStep(5.000000000000000);

        horizontalLayout_17->addWidget(doubleSpinBox_c3);


        verticalLayout_11->addLayout(horizontalLayout_17);


        horizontalLayout_34->addLayout(verticalLayout_11);

        widget_resampleCurvePlot = new MiniCurvePlot(groupBox_resampling);
        widget_resampleCurvePlot->setObjectName(QString::fromUtf8("widget_resampleCurvePlot"));
        sizePolicy3.setHeightForWidth(widget_resampleCurvePlot->sizePolicy().hasHeightForWidth());
        widget_resampleCurvePlot->setSizePolicy(sizePolicy3);
        widget_resampleCurvePlot->setMinimumSize(QSize(150, 0));

        horizontalLayout_34->addWidget(widget_resampleCurvePlot);


        verticalLayout_7->addLayout(horizontalLayout_34);


        verticalLayout_10->addWidget(groupBox_resampling);

        groupBox_dispersionCompensation = new QGroupBox(scrollAreaWidgetContents);
        groupBox_dispersionCompensation->setObjectName(QString::fromUtf8("groupBox_dispersionCompensation"));
        groupBox_dispersionCompensation->setEnabled(true);
        groupBox_dispersionCompensation->setFlat(false);
        groupBox_dispersionCompensation->setCheckable(true);
        horizontalLayout_32 = new QHBoxLayout(groupBox_dispersionCompensation);
        horizontalLayout_32->setSpacing(3);
        horizontalLayout_32->setObjectName(QString::fromUtf8("horizontalLayout_32"));
        horizontalLayout_32->setContentsMargins(3, 3, 3, 3);
        verticalLayout_12 = new QVBoxLayout();
        verticalLayout_12->setSpacing(0);
        verticalLayout_12->setObjectName(QString::fromUtf8("verticalLayout_12"));
        horizontalLayout_18 = new QHBoxLayout();
        horizontalLayout_18->setObjectName(QString::fromUtf8("horizontalLayout_18"));
        label_16 = new QLabel(groupBox_dispersionCompensation);
        label_16->setObjectName(QString::fromUtf8("label_16"));

        horizontalLayout_18->addWidget(label_16);

        doubleSpinBox_d0 = new QDoubleSpinBox(groupBox_dispersionCompensation);
        doubleSpinBox_d0->setObjectName(QString::fromUtf8("doubleSpinBox_d0"));
        sizePolicy5.setHeightForWidth(doubleSpinBox_d0->sizePolicy().hasHeightForWidth());
        doubleSpinBox_d0->setSizePolicy(sizePolicy5);
        doubleSpinBox_d0->setMinimumSize(QSize(100, 0));
        doubleSpinBox_d0->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_d0->setDecimals(6);
        doubleSpinBox_d0->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_d0->setMaximum(10000000000000000.000000000000000);
        doubleSpinBox_d0->setSingleStep(0.500000000000000);

        horizontalLayout_18->addWidget(doubleSpinBox_d0);


        verticalLayout_12->addLayout(horizontalLayout_18);

        horizontalLayout_19 = new QHBoxLayout();
        horizontalLayout_19->setObjectName(QString::fromUtf8("horizontalLayout_19"));
        label_17 = new QLabel(groupBox_dispersionCompensation);
        label_17->setObjectName(QString::fromUtf8("label_17"));

        horizontalLayout_19->addWidget(label_17);

        doubleSpinBox_d1 = new QDoubleSpinBox(groupBox_dispersionCompensation);
        doubleSpinBox_d1->setObjectName(QString::fromUtf8("doubleSpinBox_d1"));
        sizePolicy5.setHeightForWidth(doubleSpinBox_d1->sizePolicy().hasHeightForWidth());
        doubleSpinBox_d1->setSizePolicy(sizePolicy5);
        doubleSpinBox_d1->setMinimumSize(QSize(40, 0));
        doubleSpinBox_d1->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_d1->setDecimals(6);
        doubleSpinBox_d1->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_d1->setMaximum(10000000000000000.000000000000000);
        doubleSpinBox_d1->setSingleStep(0.500000000000000);

        horizontalLayout_19->addWidget(doubleSpinBox_d1);


        verticalLayout_12->addLayout(horizontalLayout_19);

        horizontalLayout_20 = new QHBoxLayout();
        horizontalLayout_20->setObjectName(QString::fromUtf8("horizontalLayout_20"));
        label_18 = new QLabel(groupBox_dispersionCompensation);
        label_18->setObjectName(QString::fromUtf8("label_18"));

        horizontalLayout_20->addWidget(label_18);

        doubleSpinBox_d2 = new QDoubleSpinBox(groupBox_dispersionCompensation);
        doubleSpinBox_d2->setObjectName(QString::fromUtf8("doubleSpinBox_d2"));
        sizePolicy5.setHeightForWidth(doubleSpinBox_d2->sizePolicy().hasHeightForWidth());
        doubleSpinBox_d2->setSizePolicy(sizePolicy5);
        doubleSpinBox_d2->setMinimumSize(QSize(40, 0));
        doubleSpinBox_d2->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_d2->setDecimals(6);
        doubleSpinBox_d2->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_d2->setMaximum(10000000000000000.000000000000000);
        doubleSpinBox_d2->setSingleStep(0.500000000000000);
        doubleSpinBox_d2->setValue(0.000000000000000);

        horizontalLayout_20->addWidget(doubleSpinBox_d2);


        verticalLayout_12->addLayout(horizontalLayout_20);

        horizontalLayout_21 = new QHBoxLayout();
        horizontalLayout_21->setObjectName(QString::fromUtf8("horizontalLayout_21"));
        label_19 = new QLabel(groupBox_dispersionCompensation);
        label_19->setObjectName(QString::fromUtf8("label_19"));

        horizontalLayout_21->addWidget(label_19);

        doubleSpinBox_d3 = new QDoubleSpinBox(groupBox_dispersionCompensation);
        doubleSpinBox_d3->setObjectName(QString::fromUtf8("doubleSpinBox_d3"));
        sizePolicy5.setHeightForWidth(doubleSpinBox_d3->sizePolicy().hasHeightForWidth());
        doubleSpinBox_d3->setSizePolicy(sizePolicy5);
        doubleSpinBox_d3->setMinimumSize(QSize(40, 0));
        doubleSpinBox_d3->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_d3->setDecimals(6);
        doubleSpinBox_d3->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_d3->setMaximum(10000000000000000.000000000000000);
        doubleSpinBox_d3->setSingleStep(0.500000000000000);

        horizontalLayout_21->addWidget(doubleSpinBox_d3);


        verticalLayout_12->addLayout(horizontalLayout_21);


        horizontalLayout_32->addLayout(verticalLayout_12);

        widget_dispersionCurvePlot = new MiniCurvePlot(groupBox_dispersionCompensation);
        widget_dispersionCurvePlot->setObjectName(QString::fromUtf8("widget_dispersionCurvePlot"));
        sizePolicy3.setHeightForWidth(widget_dispersionCurvePlot->sizePolicy().hasHeightForWidth());
        widget_dispersionCurvePlot->setSizePolicy(sizePolicy3);
        widget_dispersionCurvePlot->setMinimumSize(QSize(150, 0));

        horizontalLayout_32->addWidget(widget_dispersionCurvePlot);


        verticalLayout_10->addWidget(groupBox_dispersionCompensation);

        groupBox_windowing = new QGroupBox(scrollAreaWidgetContents);
        groupBox_windowing->setObjectName(QString::fromUtf8("groupBox_windowing"));
        groupBox_windowing->setEnabled(true);
        groupBox_windowing->setFlat(false);
        groupBox_windowing->setCheckable(true);
        horizontalLayout_33 = new QHBoxLayout(groupBox_windowing);
        horizontalLayout_33->setSpacing(3);
        horizontalLayout_33->setObjectName(QString::fromUtf8("horizontalLayout_33"));
        horizontalLayout_33->setContentsMargins(3, 3, 3, 3);
        verticalLayout_13 = new QVBoxLayout();
        verticalLayout_13->setSpacing(0);
        verticalLayout_13->setObjectName(QString::fromUtf8("verticalLayout_13"));
        horizontalLayout_24 = new QHBoxLayout();
        horizontalLayout_24->setObjectName(QString::fromUtf8("horizontalLayout_24"));
        label_22 = new QLabel(groupBox_windowing);
        label_22->setObjectName(QString::fromUtf8("label_22"));

        horizontalLayout_24->addWidget(label_22);

        comboBox_windowType = new QComboBox(groupBox_windowing);
        comboBox_windowType->setObjectName(QString::fromUtf8("comboBox_windowType"));

        horizontalLayout_24->addWidget(comboBox_windowType);


        verticalLayout_13->addLayout(horizontalLayout_24);

        horizontalLayout_23 = new QHBoxLayout();
        horizontalLayout_23->setObjectName(QString::fromUtf8("horizontalLayout_23"));
        label_23 = new QLabel(groupBox_windowing);
        label_23->setObjectName(QString::fromUtf8("label_23"));

        horizontalLayout_23->addWidget(label_23);

        doubleSpinBox_windowFillFactor = new QDoubleSpinBox(groupBox_windowing);
        doubleSpinBox_windowFillFactor->setObjectName(QString::fromUtf8("doubleSpinBox_windowFillFactor"));
        QSizePolicy sizePolicy7(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy7.setHorizontalStretch(0);
        sizePolicy7.setVerticalStretch(0);
        sizePolicy7.setHeightForWidth(doubleSpinBox_windowFillFactor->sizePolicy().hasHeightForWidth());
        doubleSpinBox_windowFillFactor->setSizePolicy(sizePolicy7);
        doubleSpinBox_windowFillFactor->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_windowFillFactor->setDecimals(2);
        doubleSpinBox_windowFillFactor->setMinimum(0.000000000000000);
        doubleSpinBox_windowFillFactor->setMaximum(10.000000000000000);
        doubleSpinBox_windowFillFactor->setSingleStep(0.010000000000000);
        doubleSpinBox_windowFillFactor->setValue(0.900000000000000);

        horizontalLayout_23->addWidget(doubleSpinBox_windowFillFactor);


        verticalLayout_13->addLayout(horizontalLayout_23);

        horizontalLayout_22 = new QHBoxLayout();
        horizontalLayout_22->setObjectName(QString::fromUtf8("horizontalLayout_22"));
        label_24 = new QLabel(groupBox_windowing);
        label_24->setObjectName(QString::fromUtf8("label_24"));

        horizontalLayout_22->addWidget(label_24);

        doubleSpinBox_windowCenterPosition = new QDoubleSpinBox(groupBox_windowing);
        doubleSpinBox_windowCenterPosition->setObjectName(QString::fromUtf8("doubleSpinBox_windowCenterPosition"));
        sizePolicy7.setHeightForWidth(doubleSpinBox_windowCenterPosition->sizePolicy().hasHeightForWidth());
        doubleSpinBox_windowCenterPosition->setSizePolicy(sizePolicy7);
        doubleSpinBox_windowCenterPosition->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_windowCenterPosition->setDecimals(2);
        doubleSpinBox_windowCenterPosition->setMinimum(0.000000000000000);
        doubleSpinBox_windowCenterPosition->setMaximum(1.000000000000000);
        doubleSpinBox_windowCenterPosition->setSingleStep(0.010000000000000);
        doubleSpinBox_windowCenterPosition->setValue(0.500000000000000);

        horizontalLayout_22->addWidget(doubleSpinBox_windowCenterPosition);


        verticalLayout_13->addLayout(horizontalLayout_22);


        horizontalLayout_33->addLayout(verticalLayout_13);

        widget_windowCurvePlot = new MiniCurvePlot(groupBox_windowing);
        widget_windowCurvePlot->setObjectName(QString::fromUtf8("widget_windowCurvePlot"));
        sizePolicy3.setHeightForWidth(widget_windowCurvePlot->sizePolicy().hasHeightForWidth());
        widget_windowCurvePlot->setSizePolicy(sizePolicy3);
        widget_windowCurvePlot->setMinimumSize(QSize(100, 0));

        horizontalLayout_33->addWidget(widget_windowCurvePlot);


        verticalLayout_10->addWidget(groupBox_windowing);

        groupBox_fixedPatternNoiseRemoval = new QGroupBox(scrollAreaWidgetContents);
        groupBox_fixedPatternNoiseRemoval->setObjectName(QString::fromUtf8("groupBox_fixedPatternNoiseRemoval"));
        groupBox_fixedPatternNoiseRemoval->setEnabled(true);
        groupBox_fixedPatternNoiseRemoval->setFlat(false);
        groupBox_fixedPatternNoiseRemoval->setCheckable(true);
        groupBox_fixedPatternNoiseRemoval->setChecked(false);
        verticalLayout_17 = new QVBoxLayout(groupBox_fixedPatternNoiseRemoval);
        verticalLayout_17->setSpacing(0);
        verticalLayout_17->setObjectName(QString::fromUtf8("verticalLayout_17"));
        verticalLayout_17->setContentsMargins(3, 3, 3, 3);
        horizontalLayout_29 = new QHBoxLayout();
        horizontalLayout_29->setSpacing(0);
        horizontalLayout_29->setObjectName(QString::fromUtf8("horizontalLayout_29"));
        label_28 = new QLabel(groupBox_fixedPatternNoiseRemoval);
        label_28->setObjectName(QString::fromUtf8("label_28"));
        label_28->setScaledContents(false);
        label_28->setWordWrap(false);

        horizontalLayout_29->addWidget(label_28);

        spinBox_bscansFixedNoise = new QSpinBox(groupBox_fixedPatternNoiseRemoval);
        spinBox_bscansFixedNoise->setObjectName(QString::fromUtf8("spinBox_bscansFixedNoise"));
        sizePolicy5.setHeightForWidth(spinBox_bscansFixedNoise->sizePolicy().hasHeightForWidth());
        spinBox_bscansFixedNoise->setSizePolicy(sizePolicy5);
        spinBox_bscansFixedNoise->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        spinBox_bscansFixedNoise->setMinimum(1);
        spinBox_bscansFixedNoise->setMaximum(999);

        horizontalLayout_29->addWidget(spinBox_bscansFixedNoise);


        verticalLayout_17->addLayout(horizontalLayout_29);

        label_29 = new QLabel(groupBox_fixedPatternNoiseRemoval);
        label_29->setObjectName(QString::fromUtf8("label_29"));
        QFont font;
        font.setPointSize(2);
        label_29->setFont(font);
        label_29->setTextFormat(Qt::PlainText);

        verticalLayout_17->addWidget(label_29);

        label_27 = new QLabel(groupBox_fixedPatternNoiseRemoval);
        label_27->setObjectName(QString::fromUtf8("label_27"));

        verticalLayout_17->addWidget(label_27);

        horizontalLayout_28 = new QHBoxLayout();
        horizontalLayout_28->setSpacing(0);
        horizontalLayout_28->setObjectName(QString::fromUtf8("horizontalLayout_28"));
        radioButton_once = new QRadioButton(groupBox_fixedPatternNoiseRemoval);
        radioButton_once->setObjectName(QString::fromUtf8("radioButton_once"));
        sizePolicy5.setHeightForWidth(radioButton_once->sizePolicy().hasHeightForWidth());
        radioButton_once->setSizePolicy(sizePolicy5);
        radioButton_once->setChecked(true);

        horizontalLayout_28->addWidget(radioButton_once);

        horizontalSpacer_9 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_28->addItem(horizontalSpacer_9);

        pushButton_redetermine = new QPushButton(groupBox_fixedPatternNoiseRemoval);
        pushButton_redetermine->setObjectName(QString::fromUtf8("pushButton_redetermine"));
        sizePolicy5.setHeightForWidth(pushButton_redetermine->sizePolicy().hasHeightForWidth());
        pushButton_redetermine->setSizePolicy(sizePolicy5);

        horizontalLayout_28->addWidget(pushButton_redetermine);


        verticalLayout_17->addLayout(horizontalLayout_28);

        radioButton_continuously = new QRadioButton(groupBox_fixedPatternNoiseRemoval);
        radioButton_continuously->setObjectName(QString::fromUtf8("radioButton_continuously"));
        sizePolicy5.setHeightForWidth(radioButton_continuously->sizePolicy().hasHeightForWidth());
        radioButton_continuously->setSizePolicy(sizePolicy5);

        verticalLayout_17->addWidget(radioButton_continuously);


        verticalLayout_10->addWidget(groupBox_fixedPatternNoiseRemoval);

        groupBox_3 = new QGroupBox(scrollAreaWidgetContents);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        verticalLayout_6 = new QVBoxLayout(groupBox_3);
        verticalLayout_6->setSpacing(0);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        verticalLayout_6->setContentsMargins(3, 3, 3, 3);
        checkBox_logScaling = new QCheckBox(groupBox_3);
        checkBox_logScaling->setObjectName(QString::fromUtf8("checkBox_logScaling"));
        sizePolicy5.setHeightForWidth(checkBox_logScaling->sizePolicy().hasHeightForWidth());
        checkBox_logScaling->setSizePolicy(sizePolicy5);

        verticalLayout_6->addWidget(checkBox_logScaling);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(0);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        label_4 = new QLabel(groupBox_3);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout_6->addWidget(label_4);

        doubleSpinBox_signalMax = new QDoubleSpinBox(groupBox_3);
        doubleSpinBox_signalMax->setObjectName(QString::fromUtf8("doubleSpinBox_signalMax"));
        doubleSpinBox_signalMax->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_signalMax->setDecimals(2);
        doubleSpinBox_signalMax->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_signalMax->setMaximum(10000000000000000.000000000000000);
        doubleSpinBox_signalMax->setSingleStep(5.000000000000000);

        horizontalLayout_6->addWidget(doubleSpinBox_signalMax);


        verticalLayout_6->addLayout(horizontalLayout_6);

        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        label_5 = new QLabel(groupBox_3);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        horizontalLayout_7->addWidget(label_5);

        doubleSpinBox_signalMin = new QDoubleSpinBox(groupBox_3);
        doubleSpinBox_signalMin->setObjectName(QString::fromUtf8("doubleSpinBox_signalMin"));
        doubleSpinBox_signalMin->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_signalMin->setDecimals(2);
        doubleSpinBox_signalMin->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_signalMin->setMaximum(10000000000000000.000000000000000);
        doubleSpinBox_signalMin->setSingleStep(5.000000000000000);

        horizontalLayout_7->addWidget(doubleSpinBox_signalMin);


        verticalLayout_6->addLayout(horizontalLayout_7);

        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        label_6 = new QLabel(groupBox_3);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        horizontalLayout_8->addWidget(label_6);

        doubleSpinBox_signalMultiplicator = new QDoubleSpinBox(groupBox_3);
        doubleSpinBox_signalMultiplicator->setObjectName(QString::fromUtf8("doubleSpinBox_signalMultiplicator"));
        doubleSpinBox_signalMultiplicator->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_signalMultiplicator->setDecimals(2);
        doubleSpinBox_signalMultiplicator->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_signalMultiplicator->setMaximum(10000000000000000.000000000000000);
        doubleSpinBox_signalMultiplicator->setSingleStep(0.100000000000000);
        doubleSpinBox_signalMultiplicator->setValue(1.000000000000000);

        horizontalLayout_8->addWidget(doubleSpinBox_signalMultiplicator);


        verticalLayout_6->addLayout(horizontalLayout_8);

        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setSpacing(0);
        horizontalLayout_9->setObjectName(QString::fromUtf8("horizontalLayout_9"));
        label_7 = new QLabel(groupBox_3);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        horizontalLayout_9->addWidget(label_7);

        doubleSpinBox_signalAddend = new QDoubleSpinBox(groupBox_3);
        doubleSpinBox_signalAddend->setObjectName(QString::fromUtf8("doubleSpinBox_signalAddend"));
        doubleSpinBox_signalAddend->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_signalAddend->setDecimals(2);
        doubleSpinBox_signalAddend->setMinimum(-10000000000000000.000000000000000);
        doubleSpinBox_signalAddend->setMaximum(10000000000000000.000000000000000);
        doubleSpinBox_signalAddend->setSingleStep(0.100000000000000);

        horizontalLayout_9->addWidget(doubleSpinBox_signalAddend);


        verticalLayout_6->addLayout(horizontalLayout_9);


        verticalLayout_10->addWidget(groupBox_3);

        groupBox_postProcessBackgroundRemoval = new QGroupBox(scrollAreaWidgetContents);
        groupBox_postProcessBackgroundRemoval->setObjectName(QString::fromUtf8("groupBox_postProcessBackgroundRemoval"));
        groupBox_postProcessBackgroundRemoval->setCheckable(true);
        groupBox_postProcessBackgroundRemoval->setChecked(false);
        horizontalLayout_38 = new QHBoxLayout(groupBox_postProcessBackgroundRemoval);
        horizontalLayout_38->setSpacing(3);
        horizontalLayout_38->setObjectName(QString::fromUtf8("horizontalLayout_38"));
        horizontalLayout_38->setContentsMargins(3, 3, 3, 3);
        verticalLayout_21 = new QVBoxLayout();
        verticalLayout_21->setSpacing(0);
        verticalLayout_21->setObjectName(QString::fromUtf8("verticalLayout_21"));
        horizontalLayout_36 = new QHBoxLayout();
        horizontalLayout_36->setSpacing(3);
        horizontalLayout_36->setObjectName(QString::fromUtf8("horizontalLayout_36"));
        pushButton_postProcRec = new QPushButton(groupBox_postProcessBackgroundRemoval);
        pushButton_postProcRec->setObjectName(QString::fromUtf8("pushButton_postProcRec"));
        QSizePolicy sizePolicy8(QSizePolicy::Maximum, QSizePolicy::Fixed);
        sizePolicy8.setHorizontalStretch(0);
        sizePolicy8.setVerticalStretch(0);
        sizePolicy8.setHeightForWidth(pushButton_postProcRec->sizePolicy().hasHeightForWidth());
        pushButton_postProcRec->setSizePolicy(sizePolicy8);
        pushButton_postProcRec->setMinimumSize(QSize(45, 0));

        horizontalLayout_36->addWidget(pushButton_postProcRec);

        pushButton_postProcSave = new QPushButton(groupBox_postProcessBackgroundRemoval);
        pushButton_postProcSave->setObjectName(QString::fromUtf8("pushButton_postProcSave"));
        sizePolicy8.setHeightForWidth(pushButton_postProcSave->sizePolicy().hasHeightForWidth());
        pushButton_postProcSave->setSizePolicy(sizePolicy8);
        pushButton_postProcSave->setMinimumSize(QSize(45, 0));

        horizontalLayout_36->addWidget(pushButton_postProcSave);

        pushButton_postProcLoad = new QPushButton(groupBox_postProcessBackgroundRemoval);
        pushButton_postProcLoad->setObjectName(QString::fromUtf8("pushButton_postProcLoad"));
        sizePolicy8.setHeightForWidth(pushButton_postProcLoad->sizePolicy().hasHeightForWidth());
        pushButton_postProcLoad->setSizePolicy(sizePolicy8);
        pushButton_postProcLoad->setMinimumSize(QSize(45, 0));

        horizontalLayout_36->addWidget(pushButton_postProcLoad);


        verticalLayout_21->addLayout(horizontalLayout_36);

        horizontalLayout_35 = new QHBoxLayout();
        horizontalLayout_35->setObjectName(QString::fromUtf8("horizontalLayout_35"));
        label_8 = new QLabel(groupBox_postProcessBackgroundRemoval);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        sizePolicy4.setHeightForWidth(label_8->sizePolicy().hasHeightForWidth());
        label_8->setSizePolicy(sizePolicy4);

        horizontalLayout_35->addWidget(label_8);

        horizontalSpacer_13 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_35->addItem(horizontalSpacer_13);

        doubleSpinBox_postProcessBackgroundWeight = new QDoubleSpinBox(groupBox_postProcessBackgroundRemoval);
        doubleSpinBox_postProcessBackgroundWeight->setObjectName(QString::fromUtf8("doubleSpinBox_postProcessBackgroundWeight"));
        QSizePolicy sizePolicy9(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy9.setHorizontalStretch(0);
        sizePolicy9.setVerticalStretch(0);
        sizePolicy9.setHeightForWidth(doubleSpinBox_postProcessBackgroundWeight->sizePolicy().hasHeightForWidth());
        doubleSpinBox_postProcessBackgroundWeight->setSizePolicy(sizePolicy9);
        doubleSpinBox_postProcessBackgroundWeight->setMinimumSize(QSize(80, 0));
        doubleSpinBox_postProcessBackgroundWeight->setMaximumSize(QSize(80, 16777215));
        doubleSpinBox_postProcessBackgroundWeight->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_postProcessBackgroundWeight->setMaximum(1000.000000000000000);
        doubleSpinBox_postProcessBackgroundWeight->setSingleStep(0.010000000000000);
        doubleSpinBox_postProcessBackgroundWeight->setValue(1.000000000000000);

        horizontalLayout_35->addWidget(doubleSpinBox_postProcessBackgroundWeight);


        verticalLayout_21->addLayout(horizontalLayout_35);

        horizontalLayout_37 = new QHBoxLayout();
        horizontalLayout_37->setObjectName(QString::fromUtf8("horizontalLayout_37"));
        label_9 = new QLabel(groupBox_postProcessBackgroundRemoval);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        sizePolicy4.setHeightForWidth(label_9->sizePolicy().hasHeightForWidth());
        label_9->setSizePolicy(sizePolicy4);

        horizontalLayout_37->addWidget(label_9);

        horizontalSpacer_12 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_37->addItem(horizontalSpacer_12);

        doubleSpinBox_postProcessBackgroundOffset = new QDoubleSpinBox(groupBox_postProcessBackgroundRemoval);
        doubleSpinBox_postProcessBackgroundOffset->setObjectName(QString::fromUtf8("doubleSpinBox_postProcessBackgroundOffset"));
        sizePolicy9.setHeightForWidth(doubleSpinBox_postProcessBackgroundOffset->sizePolicy().hasHeightForWidth());
        doubleSpinBox_postProcessBackgroundOffset->setSizePolicy(sizePolicy9);
        doubleSpinBox_postProcessBackgroundOffset->setMinimumSize(QSize(80, 0));
        doubleSpinBox_postProcessBackgroundOffset->setMaximumSize(QSize(80, 16777215));
        doubleSpinBox_postProcessBackgroundOffset->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        doubleSpinBox_postProcessBackgroundOffset->setMinimum(-1000.000000000000000);
        doubleSpinBox_postProcessBackgroundOffset->setMaximum(1000.000000000000000);
        doubleSpinBox_postProcessBackgroundOffset->setSingleStep(0.010000000000000);

        horizontalLayout_37->addWidget(doubleSpinBox_postProcessBackgroundOffset);


        verticalLayout_21->addLayout(horizontalLayout_37);


        horizontalLayout_38->addLayout(verticalLayout_21);

        widget_postProcessBackgroundPlot = new MiniCurvePlot(groupBox_postProcessBackgroundRemoval);
        widget_postProcessBackgroundPlot->setObjectName(QString::fromUtf8("widget_postProcessBackgroundPlot"));
        sizePolicy3.setHeightForWidth(widget_postProcessBackgroundPlot->sizePolicy().hasHeightForWidth());
        widget_postProcessBackgroundPlot->setSizePolicy(sizePolicy3);
        widget_postProcessBackgroundPlot->setMinimumSize(QSize(146, 0));
        widget_postProcessBackgroundPlot->setMaximumSize(QSize(146, 16777215));

        horizontalLayout_38->addWidget(widget_postProcessBackgroundPlot);


        verticalLayout_10->addWidget(groupBox_postProcessBackgroundRemoval);

        groupBox_streaming = new QGroupBox(scrollAreaWidgetContents);
        groupBox_streaming->setObjectName(QString::fromUtf8("groupBox_streaming"));
        groupBox_streaming->setCheckable(true);
        verticalLayout = new QVBoxLayout(groupBox_streaming);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(3, 3, 3, 3);
        horizontalLayout_26 = new QHBoxLayout();
        horizontalLayout_26->setSpacing(6);
        horizontalLayout_26->setObjectName(QString::fromUtf8("horizontalLayout_26"));
        label_26 = new QLabel(groupBox_streaming);
        label_26->setObjectName(QString::fromUtf8("label_26"));

        horizontalLayout_26->addWidget(label_26);

        spinBox_streamingBuffersToSkip = new QSpinBox(groupBox_streaming);
        spinBox_streamingBuffersToSkip->setObjectName(QString::fromUtf8("spinBox_streamingBuffersToSkip"));
        spinBox_streamingBuffersToSkip->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_26->addWidget(spinBox_streamingBuffersToSkip);


        verticalLayout->addLayout(horizontalLayout_26);


        verticalLayout_10->addWidget(groupBox_streaming);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_10->addItem(verticalSpacer_3);

        scrollArea->setWidget(scrollAreaWidgetContents);

        verticalLayout_20->addWidget(scrollArea);

        groupBox_info = new QGroupBox(tab);
        groupBox_info->setObjectName(QString::fromUtf8("groupBox_info"));
        groupBox_info->setMaximumSize(QSize(16777215, 100));
        groupBox_info->setContextMenuPolicy(Qt::ActionsContextMenu);
        verticalLayout_8 = new QVBoxLayout(groupBox_info);
        verticalLayout_8->setSpacing(0);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        verticalLayout_8->setContentsMargins(3, 0, 3, 0);
        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        label_name_volumesPerSecond = new QLabel(groupBox_info);
        label_name_volumesPerSecond->setObjectName(QString::fromUtf8("label_name_volumesPerSecond"));
        QSizePolicy sizePolicy10(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
        sizePolicy10.setHorizontalStretch(0);
        sizePolicy10.setVerticalStretch(0);
        sizePolicy10.setHeightForWidth(label_name_volumesPerSecond->sizePolicy().hasHeightForWidth());
        label_name_volumesPerSecond->setSizePolicy(sizePolicy10);
        label_name_volumesPerSecond->setTextInteractionFlags(Qt::NoTextInteraction);

        horizontalLayout_5->addWidget(label_name_volumesPerSecond);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(horizontalSpacer_3);

        label_volumesPerSecond = new QLabel(groupBox_info);
        label_volumesPerSecond->setObjectName(QString::fromUtf8("label_volumesPerSecond"));
        label_volumesPerSecond->setFrameShape(QFrame::NoFrame);
        label_volumesPerSecond->setLineWidth(0);
        label_volumesPerSecond->setTextInteractionFlags(Qt::NoTextInteraction);

        horizontalLayout_5->addWidget(label_volumesPerSecond);


        verticalLayout_8->addLayout(horizontalLayout_5);

        horizontalLayout_30 = new QHBoxLayout();
        horizontalLayout_30->setObjectName(QString::fromUtf8("horizontalLayout_30"));
        label_name_buffersPerSecond = new QLabel(groupBox_info);
        label_name_buffersPerSecond->setObjectName(QString::fromUtf8("label_name_buffersPerSecond"));
        sizePolicy10.setHeightForWidth(label_name_buffersPerSecond->sizePolicy().hasHeightForWidth());
        label_name_buffersPerSecond->setSizePolicy(sizePolicy10);
        label_name_buffersPerSecond->setTextInteractionFlags(Qt::NoTextInteraction);

        horizontalLayout_30->addWidget(label_name_buffersPerSecond);

        horizontalSpacer_8 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_30->addItem(horizontalSpacer_8);

        label_buffersPerSecond = new QLabel(groupBox_info);
        label_buffersPerSecond->setObjectName(QString::fromUtf8("label_buffersPerSecond"));
        label_buffersPerSecond->setFrameShape(QFrame::NoFrame);
        label_buffersPerSecond->setLineWidth(0);
        label_buffersPerSecond->setTextInteractionFlags(Qt::NoTextInteraction);

        horizontalLayout_30->addWidget(label_buffersPerSecond);


        verticalLayout_8->addLayout(horizontalLayout_30);

        horizontalLayout_12 = new QHBoxLayout();
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        label_name_bscansPerSecond = new QLabel(groupBox_info);
        label_name_bscansPerSecond->setObjectName(QString::fromUtf8("label_name_bscansPerSecond"));
        sizePolicy10.setHeightForWidth(label_name_bscansPerSecond->sizePolicy().hasHeightForWidth());
        label_name_bscansPerSecond->setSizePolicy(sizePolicy10);
        label_name_bscansPerSecond->setTextInteractionFlags(Qt::NoTextInteraction);

        horizontalLayout_12->addWidget(label_name_bscansPerSecond);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_12->addItem(horizontalSpacer_6);

        label_bscansPerSecond = new QLabel(groupBox_info);
        label_bscansPerSecond->setObjectName(QString::fromUtf8("label_bscansPerSecond"));
        label_bscansPerSecond->setFrameShape(QFrame::NoFrame);
        label_bscansPerSecond->setLineWidth(0);
        label_bscansPerSecond->setTextInteractionFlags(Qt::NoTextInteraction);

        horizontalLayout_12->addWidget(label_bscansPerSecond);


        verticalLayout_8->addLayout(horizontalLayout_12);

        horizontalLayout_13 = new QHBoxLayout();
        horizontalLayout_13->setObjectName(QString::fromUtf8("horizontalLayout_13"));
        label_name_ascansPerSecond = new QLabel(groupBox_info);
        label_name_ascansPerSecond->setObjectName(QString::fromUtf8("label_name_ascansPerSecond"));
        sizePolicy10.setHeightForWidth(label_name_ascansPerSecond->sizePolicy().hasHeightForWidth());
        label_name_ascansPerSecond->setSizePolicy(sizePolicy10);
        label_name_ascansPerSecond->setTextInteractionFlags(Qt::NoTextInteraction);

        horizontalLayout_13->addWidget(label_name_ascansPerSecond);

        horizontalSpacer_7 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_13->addItem(horizontalSpacer_7);

        label_ascansPerSecond = new QLabel(groupBox_info);
        label_ascansPerSecond->setObjectName(QString::fromUtf8("label_ascansPerSecond"));
        label_ascansPerSecond->setFrameShape(QFrame::NoFrame);
        label_ascansPerSecond->setLineWidth(0);
        label_ascansPerSecond->setTextInteractionFlags(Qt::NoTextInteraction);

        horizontalLayout_13->addWidget(label_ascansPerSecond);


        verticalLayout_8->addLayout(horizontalLayout_13);

        horizontalLayout_10 = new QHBoxLayout();
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        label_name_bufferSize = new QLabel(groupBox_info);
        label_name_bufferSize->setObjectName(QString::fromUtf8("label_name_bufferSize"));
        sizePolicy10.setHeightForWidth(label_name_bufferSize->sizePolicy().hasHeightForWidth());
        label_name_bufferSize->setSizePolicy(sizePolicy10);
        label_name_bufferSize->setTextInteractionFlags(Qt::NoTextInteraction);

        horizontalLayout_10->addWidget(label_name_bufferSize);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_10->addItem(horizontalSpacer_4);

        label_bufferSize = new QLabel(groupBox_info);
        label_bufferSize->setObjectName(QString::fromUtf8("label_bufferSize"));
        label_bufferSize->setFrameShape(QFrame::NoFrame);
        label_bufferSize->setFrameShadow(QFrame::Plain);
        label_bufferSize->setLineWidth(0);
        label_bufferSize->setTextInteractionFlags(Qt::NoTextInteraction);

        horizontalLayout_10->addWidget(label_bufferSize);


        verticalLayout_8->addLayout(horizontalLayout_10);

        horizontalLayout_11 = new QHBoxLayout();
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        label_name_dataThroughput = new QLabel(groupBox_info);
        label_name_dataThroughput->setObjectName(QString::fromUtf8("label_name_dataThroughput"));
        label_name_dataThroughput->setTextInteractionFlags(Qt::NoTextInteraction);

        horizontalLayout_11->addWidget(label_name_dataThroughput);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_11->addItem(horizontalSpacer_5);

        label_dataThroughput = new QLabel(groupBox_info);
        label_dataThroughput->setObjectName(QString::fromUtf8("label_dataThroughput"));
        label_dataThroughput->setTextInteractionFlags(Qt::NoTextInteraction);

        horizontalLayout_11->addWidget(label_dataThroughput);


        verticalLayout_8->addLayout(horizontalLayout_11);


        verticalLayout_20->addWidget(groupBox_info);

        tabWidget->addTab(tab, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QString::fromUtf8("tab_3"));
        sizePolicy3.setHeightForWidth(tab_3->sizePolicy().hasHeightForWidth());
        tab_3->setSizePolicy(sizePolicy3);
        verticalLayout_16 = new QVBoxLayout(tab_3);
        verticalLayout_16->setObjectName(QString::fromUtf8("verticalLayout_16"));
        verticalLayout_16->setContentsMargins(3, 9, 3, -1);
        scrollArea_2 = new QScrollArea(tab_3);
        scrollArea_2->setObjectName(QString::fromUtf8("scrollArea_2"));
        sizePolicy3.setHeightForWidth(scrollArea_2->sizePolicy().hasHeightForWidth());
        scrollArea_2->setSizePolicy(sizePolicy3);
        scrollArea_2->setFrameShape(QFrame::NoFrame);
        scrollArea_2->setFrameShadow(QFrame::Plain);
        scrollArea_2->setLineWidth(0);
        scrollArea_2->setSizeAdjustPolicy(QAbstractScrollArea::AdjustIgnored);
        scrollArea_2->setWidgetResizable(true);
        scrollAreaWidgetContents_2 = new QWidget();
        scrollAreaWidgetContents_2->setObjectName(QString::fromUtf8("scrollAreaWidgetContents_2"));
        scrollAreaWidgetContents_2->setGeometry(QRect(0, 0, 324, 479));
        sizePolicy4.setHeightForWidth(scrollAreaWidgetContents_2->sizePolicy().hasHeightForWidth());
        scrollAreaWidgetContents_2->setSizePolicy(sizePolicy4);
        verticalLayout_22 = new QVBoxLayout(scrollAreaWidgetContents_2);
        verticalLayout_22->setObjectName(QString::fromUtf8("verticalLayout_22"));
        verticalLayout_22->setContentsMargins(0, 0, 3, 0);
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(3);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        label = new QLabel(scrollAreaWidgetContents_2);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_3->addWidget(label);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_2);


        verticalLayout_2->addLayout(horizontalLayout_3);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        lineEdit_saveFolder = new QLineEdit(scrollAreaWidgetContents_2);
        lineEdit_saveFolder->setObjectName(QString::fromUtf8("lineEdit_saveFolder"));
        lineEdit_saveFolder->setReadOnly(false);
        lineEdit_saveFolder->setCursorMoveStyle(Qt::VisualMoveStyle);
        lineEdit_saveFolder->setClearButtonEnabled(false);

        horizontalLayout_2->addWidget(lineEdit_saveFolder);

        toolButton_recPath = new QToolButton(scrollAreaWidgetContents_2);
        toolButton_recPath->setObjectName(QString::fromUtf8("toolButton_recPath"));

        horizontalLayout_2->addWidget(toolButton_recPath);


        verticalLayout_2->addLayout(horizontalLayout_2);


        verticalLayout_22->addLayout(verticalLayout_2);

        groupBox = new QGroupBox(scrollAreaWidgetContents_2);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        verticalLayout_3 = new QVBoxLayout(groupBox);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(3, 3, 3, 3);
        checkBox_recordScreenshots = new QCheckBox(groupBox);
        checkBox_recordScreenshots->setObjectName(QString::fromUtf8("checkBox_recordScreenshots"));

        verticalLayout_3->addWidget(checkBox_recordScreenshots);

        checkBox_recordRawBuffers = new QCheckBox(groupBox);
        checkBox_recordRawBuffers->setObjectName(QString::fromUtf8("checkBox_recordRawBuffers"));

        verticalLayout_3->addWidget(checkBox_recordRawBuffers);

        checkBox_recordProcessedBuffers = new QCheckBox(groupBox);
        checkBox_recordProcessedBuffers->setObjectName(QString::fromUtf8("checkBox_recordProcessedBuffers"));

        verticalLayout_3->addWidget(checkBox_recordProcessedBuffers);


        verticalLayout_22->addWidget(groupBox);

        groupBox_2 = new QGroupBox(scrollAreaWidgetContents_2);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        verticalLayout_4 = new QVBoxLayout(groupBox_2);
        verticalLayout_4->setSpacing(0);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(3, 3, 3, 3);
        checkBox_startWithFirstBuffer = new QCheckBox(groupBox_2);
        checkBox_startWithFirstBuffer->setObjectName(QString::fromUtf8("checkBox_startWithFirstBuffer"));

        verticalLayout_4->addWidget(checkBox_startWithFirstBuffer);

        checkBox_stopAfterRec = new QCheckBox(groupBox_2);
        checkBox_stopAfterRec->setObjectName(QString::fromUtf8("checkBox_stopAfterRec"));

        verticalLayout_4->addWidget(checkBox_stopAfterRec);

        checkBox_meta = new QCheckBox(groupBox_2);
        checkBox_meta->setObjectName(QString::fromUtf8("checkBox_meta"));

        verticalLayout_4->addWidget(checkBox_meta);

        checkBox_32bitfloat = new QCheckBox(groupBox_2);
        checkBox_32bitfloat->setObjectName(QString::fromUtf8("checkBox_32bitfloat"));

        verticalLayout_4->addWidget(checkBox_32bitfloat);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout_4->addWidget(label_2);

        spinBox_volumes = new QSpinBox(groupBox_2);
        spinBox_volumes->setObjectName(QString::fromUtf8("spinBox_volumes"));
        spinBox_volumes->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        spinBox_volumes->setMinimum(1);
        spinBox_volumes->setMaximum(9999);

        horizontalLayout_4->addWidget(spinBox_volumes);


        verticalLayout_4->addLayout(horizontalLayout_4);


        verticalLayout_22->addWidget(groupBox_2);

        groupBox_6 = new QGroupBox(scrollAreaWidgetContents_2);
        groupBox_6->setObjectName(QString::fromUtf8("groupBox_6"));
        sizePolicy3.setHeightForWidth(groupBox_6->sizePolicy().hasHeightForWidth());
        groupBox_6->setSizePolicy(sizePolicy3);
        verticalLayout_15 = new QVBoxLayout(groupBox_6);
        verticalLayout_15->setSpacing(3);
        verticalLayout_15->setObjectName(QString::fromUtf8("verticalLayout_15"));
        verticalLayout_15->setContentsMargins(3, 3, 3, 3);
        verticalLayout_14 = new QVBoxLayout();
        verticalLayout_14->setSpacing(3);
        verticalLayout_14->setObjectName(QString::fromUtf8("verticalLayout_14"));
        label_20 = new QLabel(groupBox_6);
        label_20->setObjectName(QString::fromUtf8("label_20"));

        verticalLayout_14->addWidget(label_20);

        lineEdit_recName = new QLineEdit(groupBox_6);
        lineEdit_recName->setObjectName(QString::fromUtf8("lineEdit_recName"));

        verticalLayout_14->addWidget(lineEdit_recName);


        verticalLayout_15->addLayout(verticalLayout_14);

        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setSpacing(3);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        label_21 = new QLabel(groupBox_6);
        label_21->setObjectName(QString::fromUtf8("label_21"));

        verticalLayout_5->addWidget(label_21);

        plainTextEdit_description = new QPlainTextEdit(groupBox_6);
        plainTextEdit_description->setObjectName(QString::fromUtf8("plainTextEdit_description"));
        sizePolicy3.setHeightForWidth(plainTextEdit_description->sizePolicy().hasHeightForWidth());
        plainTextEdit_description->setSizePolicy(sizePolicy3);
        plainTextEdit_description->setMaximumSize(QSize(16777215, 16777215));

        verticalLayout_5->addWidget(plainTextEdit_description);


        verticalLayout_15->addLayout(verticalLayout_5);


        verticalLayout_22->addWidget(groupBox_6);

        scrollArea_2->setWidget(scrollAreaWidgetContents_2);

        verticalLayout_16->addWidget(scrollArea_2);

        tabWidget->addTab(tab_3, QString());

        verticalLayout_18->addWidget(tabWidget);


        horizontalLayout_27->addWidget(widget_sidebarContent);

        pushButton_showSidebar = new QPushButton(dockWidgetContents);
        pushButton_showSidebar->setObjectName(QString::fromUtf8("pushButton_showSidebar"));
        QSizePolicy sizePolicy11(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy11.setHorizontalStretch(0);
        sizePolicy11.setVerticalStretch(0);
        sizePolicy11.setHeightForWidth(pushButton_showSidebar->sizePolicy().hasHeightForWidth());
        pushButton_showSidebar->setSizePolicy(sizePolicy11);
        pushButton_showSidebar->setMaximumSize(QSize(10, 16777215));
        pushButton_showSidebar->setSizeIncrement(QSize(0, 0));
        pushButton_showSidebar->setBaseSize(QSize(0, 0));
        QFont font1;
        font1.setBold(false);
        font1.setWeight(50);
        font1.setKerning(true);
        pushButton_showSidebar->setFont(font1);
        pushButton_showSidebar->setCursor(QCursor(Qt::PointingHandCursor));
        pushButton_showSidebar->setCheckable(false);
        pushButton_showSidebar->setFlat(true);

        horizontalLayout_27->addWidget(pushButton_showSidebar);

        Sidebar->setWidget(dockWidgetContents);

        retranslateUi(Sidebar);

        tabWidget->setCurrentIndex(0);
        pushButton_showSidebar->setDefault(false);


        QMetaObject::connectSlotsByName(Sidebar);
    } // setupUi

    void retranslateUi(QDockWidget *Sidebar)
    {
        Sidebar->setWindowTitle(QString());
        toolButton_start->setText(QString());
        toolButton_stop->setText(QString());
        toolButton_rec->setText(QString());
        toolButton_system->setText(QString());
        toolButton_settings->setText(QString());
        groupBox_rawSignalCorrection->setTitle(QCoreApplication::translate("Sidebar", "Acquisition adjustments", nullptr));
        checkBox_bitshift->setText(QCoreApplication::translate("Sidebar", "Bit shift raw sample values by 4", nullptr));
        groupBox_5->setTitle(QCoreApplication::translate("Sidebar", "Scan correction", nullptr));
        checkBox_sinusoidalScanCorrection->setText(QCoreApplication::translate("Sidebar", "Sinusoidal scan correction", nullptr));
        checkBox_bscanFlip->setText(QCoreApplication::translate("Sidebar", "Flip every second B-scan", nullptr));
        groupBox_backgroundremoval->setTitle(QCoreApplication::translate("Sidebar", "Rolling average background removal", nullptr));
        label_3->setText(QCoreApplication::translate("Sidebar", "Window size: ", nullptr));
        groupBox_resampling->setTitle(QCoreApplication::translate("Sidebar", "k-linearization", nullptr));
        label_31->setText(QCoreApplication::translate("Sidebar", "Interpolation:", nullptr));
        label_12->setText(QCoreApplication::translate("Sidebar", "c<sub>0</sub>: ", nullptr));
        label_13->setText(QCoreApplication::translate("Sidebar", "c<sub>1</sub>: ", nullptr));
        label_14->setText(QCoreApplication::translate("Sidebar", "c<sub>2</sub>: ", nullptr));
        label_15->setText(QCoreApplication::translate("Sidebar", "c<sub>3</sub>: ", nullptr));
        groupBox_dispersionCompensation->setTitle(QCoreApplication::translate("Sidebar", "Dispersion compensation", nullptr));
        label_16->setText(QCoreApplication::translate("Sidebar", "d<sub>0</sub>: ", nullptr));
        label_17->setText(QCoreApplication::translate("Sidebar", "d<sub>1</sub>: ", nullptr));
        label_18->setText(QCoreApplication::translate("Sidebar", "d<sub>2</sub>: ", nullptr));
        label_19->setText(QCoreApplication::translate("Sidebar", "d<sub>3</sub>: ", nullptr));
        groupBox_windowing->setTitle(QCoreApplication::translate("Sidebar", "Windowing", nullptr));
        label_22->setText(QCoreApplication::translate("Sidebar", "Window Type: ", nullptr));
        label_23->setText(QCoreApplication::translate("Sidebar", "Fill Factor: ", nullptr));
        label_24->setText(QCoreApplication::translate("Sidebar", "Center Position: ", nullptr));
        groupBox_fixedPatternNoiseRemoval->setTitle(QCoreApplication::translate("Sidebar", "Fixed-pattern noise removal", nullptr));
        label_28->setText(QCoreApplication::translate("Sidebar", "B-scans for noise determination: ", nullptr));
        label_29->setText(QString());
        label_27->setText(QCoreApplication::translate("Sidebar", "Determine fixed-pattern noise...", nullptr));
        radioButton_once->setText(QCoreApplication::translate("Sidebar", "once at start of measurement", nullptr));
        pushButton_redetermine->setText(QCoreApplication::translate("Sidebar", "Redetermine", nullptr));
        radioButton_continuously->setText(QCoreApplication::translate("Sidebar", "continuously", nullptr));
        groupBox_3->setTitle(QCoreApplication::translate("Sidebar", "Grayscale conversion", nullptr));
        checkBox_logScaling->setText(QCoreApplication::translate("Sidebar", "Log scaling", nullptr));
        label_4->setText(QCoreApplication::translate("Sidebar", "Max value: ", nullptr));
        label_5->setText(QCoreApplication::translate("Sidebar", "Min value: ", nullptr));
        label_6->setText(QCoreApplication::translate("Sidebar", "Multiplicator: ", nullptr));
        label_7->setText(QCoreApplication::translate("Sidebar", "Offset: ", nullptr));
        groupBox_postProcessBackgroundRemoval->setTitle(QCoreApplication::translate("Sidebar", "Post processing background removal", nullptr));
        pushButton_postProcRec->setText(QCoreApplication::translate("Sidebar", "Rec", nullptr));
        pushButton_postProcSave->setText(QCoreApplication::translate("Sidebar", "Save", nullptr));
        pushButton_postProcLoad->setText(QCoreApplication::translate("Sidebar", "Load", nullptr));
        label_8->setText(QCoreApplication::translate("Sidebar", "Weight: ", nullptr));
        label_9->setText(QCoreApplication::translate("Sidebar", "Offset: ", nullptr));
        groupBox_streaming->setTitle(QCoreApplication::translate("Sidebar", "Stream processed data to RAM", nullptr));
        label_26->setText(QCoreApplication::translate("Sidebar", "Buffers to skip:", nullptr));
        groupBox_info->setTitle(QCoreApplication::translate("Sidebar", "Info", nullptr));
        label_name_volumesPerSecond->setText(QCoreApplication::translate("Sidebar", "Volumes/second:", nullptr));
        label_volumesPerSecond->setText(QCoreApplication::translate("Sidebar", "0", nullptr));
        label_name_buffersPerSecond->setText(QCoreApplication::translate("Sidebar", "Buffers/second:", nullptr));
        label_buffersPerSecond->setText(QCoreApplication::translate("Sidebar", "0", nullptr));
        label_name_bscansPerSecond->setText(QCoreApplication::translate("Sidebar", "B-scans/second:", nullptr));
        label_bscansPerSecond->setText(QCoreApplication::translate("Sidebar", "0", nullptr));
        label_name_ascansPerSecond->setText(QCoreApplication::translate("Sidebar", "A-scans/second:", nullptr));
        label_ascansPerSecond->setText(QCoreApplication::translate("Sidebar", "0", nullptr));
        label_name_bufferSize->setText(QCoreApplication::translate("Sidebar", "Buffer size [MB]:", nullptr));
        label_bufferSize->setText(QCoreApplication::translate("Sidebar", "0", nullptr));
        label_name_dataThroughput->setText(QCoreApplication::translate("Sidebar", "Data throughput [MB/s]:", nullptr));
        label_dataThroughput->setText(QCoreApplication::translate("Sidebar", "0", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab), QCoreApplication::translate("Sidebar", "Processing", nullptr));
        label->setText(QCoreApplication::translate("Sidebar", "Recordings folder:", nullptr));
        toolButton_recPath->setText(QCoreApplication::translate("Sidebar", "...", nullptr));
        groupBox->setTitle(QCoreApplication::translate("Sidebar", "What to record", nullptr));
        checkBox_recordScreenshots->setText(QCoreApplication::translate("Sidebar", "Screenshots", nullptr));
        checkBox_recordRawBuffers->setText(QCoreApplication::translate("Sidebar", "Raw buffers", nullptr));
        checkBox_recordProcessedBuffers->setText(QCoreApplication::translate("Sidebar", "Processed buffers", nullptr));
        groupBox_2->setTitle(QCoreApplication::translate("Sidebar", "Record settings", nullptr));
        checkBox_startWithFirstBuffer->setText(QCoreApplication::translate("Sidebar", "Start recording with first buffer of volume", nullptr));
        checkBox_stopAfterRec->setText(QCoreApplication::translate("Sidebar", "Stop acquisition after record", nullptr));
        checkBox_meta->setText(QCoreApplication::translate("Sidebar", "Save meta information", nullptr));
        checkBox_32bitfloat->setText(QCoreApplication::translate("Sidebar", "Save processed buffers as 32-bit float", nullptr));
        label_2->setText(QCoreApplication::translate("Sidebar", "Buffers to record:", nullptr));
        groupBox_6->setTitle(QCoreApplication::translate("Sidebar", "Additional information", nullptr));
        label_20->setText(QCoreApplication::translate("Sidebar", "Recording name:", nullptr));
        label_21->setText(QCoreApplication::translate("Sidebar", "Description:", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QCoreApplication::translate("Sidebar", "Recording", nullptr));
#if QT_CONFIG(tooltip)
        pushButton_showSidebar->setToolTip(QCoreApplication::translate("Sidebar", "Hide sidebar", nullptr));
#endif // QT_CONFIG(tooltip)
        pushButton_showSidebar->setText(QCoreApplication::translate("Sidebar", "<", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Sidebar: public Ui_Sidebar {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SIDEBAR_H
