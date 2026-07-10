"""
集中管理 PyQt5 QSS 样式。

视觉方向：浅色、低饱和、科研演示风。界面文件只负责布局和行为，
颜色、圆角、按钮、卡片、图像画布、指标卡片等统一在这里维护。
"""

APP_STYLE = """
/* ========== Base ========== */
QMainWindow {
    background-color: #F5F7FA;
}

QWidget {
    color: #1F2937;
    font-size: 12px;
    font-family: "Microsoft YaHei";
}

QLabel {
    color: #374151;
}

QToolTip {
    background-color: #FFFFFF;
    color: #374151;
    border: 1px solid #DDE3EA;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
}

/* ========== Tabs ========== */
QTabWidget::pane {
    border: none;
    background-color: #F5F7FA;
}

QTabBar::tab {
    background-color: #EEF2F5;
    color: #5D6B7A;
    padding: 11px 28px;
    margin: 8px 3px 0px 3px;
    border: 1px solid #DDE3EA;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-size: 13px;
    font-weight: 600;
}

QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #3B82A0;
    border-color: #C9D6E2;
    border-bottom: 2px solid #FFFFFF;
}

QTabBar::tab:hover:!selected {
    background-color: #E6EDF3;
    color: #3F5F76;
}

/* ========== Cards & Sections ========== */
QFrame#pageHeader,
QFrame#panelCard,
QFrame#imageCard,
QFrame#metricPanel,
QFrame#logCard,
QFrame#statsPanel,
QFrame#noteCard {
    background-color: #FFFFFF;
    border: 1px solid #DDE3EA;
    border-radius: 10px;
}

QFrame#pageHeader {
    background-color: #FBFCFD;
}

QFrame#configPanel {
    background-color: #FFFFFF;
    border: 1px solid #DDE3EA;
    border-radius: 10px;
}

QScrollArea#workflowScrollArea,
QWidget#workflowScrollContent {
    background-color: transparent;
    border: none;
}

QFrame#workflowFooter {
    background-color: transparent;
    border: none;
}

QFrame#workflowField {
    background-color: transparent;
    border: none;
}

QFrame#imageCard {
    background-color: #FFFFFF;
}

QFrame#imageCanvasFrame {
    background-color: #F8FAFC;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
}

QFrame#metricCard,
QFrame#summaryCard,
QFrame#evalMetricCard {
    background-color: #F8FAFC;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
}

QFrame#metricCard[accent="generation"] {
    border-left: 3px solid #4F8A8B;
}

QFrame#summaryCard[accent="generation"],
QFrame#evalMetricCard[accent="generation"] {
    border-left: 3px solid #4F8A8B;
}

QFrame#summaryCard[accent="detection"],
QFrame#evalMetricCard[accent="detection"] {
    border-left: 3px solid #5D9C8A;
}

QFrame#algorithmCard {
    background-color: #F7FAFB;
    border: 1px solid #D7E1E8;
    border-radius: 8px;
}

QFrame#statusBadge {
    background-color: #F2F6F8;
    border: 1px solid #D7E1E8;
    border-radius: 8px;
}

QFrame#subtleDivider {
    background-color: #E5E7EB;
    border: none;
}

/* ========== Text Labels ========== */
QLabel#pageTitle {
    color: #1F2937;
    font-size: 20px;
    font-weight: 700;
}

QLabel#pageSubtitle {
    color: #6B7280;
    font-size: 12px;
}

QLabel#sectionTitle {
    color: #263746;
    font-size: 14px;
    font-weight: 700;
}

QLabel#sectionHint {
    color: #6B7280;
    font-size: 11px;
}

QLabel#fieldLabel {
    color: #4B5563;
    font-size: 12px;
    font-weight: 600;
}

QLabel#imageTitle {
    color: #2F4050;
    font-size: 13px;
    font-weight: 700;
}

QLabel#imageHint {
    color: #7A8794;
    font-size: 11px;
}

QLabel#flowArrow {
    color: #7B92A5;
    font-size: 28px;
    font-weight: 600;
}

QLabel#metricName,
QLabel#summaryName {
    color: #6B7280;
    font-size: 11px;
    font-weight: 600;
}

QLabel#metricValue,
QLabel#summaryValue {
    color: #2B4C5F;
    font-size: 18px;
    font-weight: 700;
}

QLabel#metricUnit {
    color: #7A8794;
    font-size: 10px;
}

QLabel#pathValue {
    color: #4F6F8F;
    font-size: 10px;
}

QLabel#statusLabel {
    color: #6B7280;
    font-size: 11px;
    font-weight: 600;
}

QLabel#statusValue {
    color: #2B4C5F;
    font-size: 12px;
    font-weight: 700;
}

QLabel#noteText {
    color: #6B7280;
    font-size: 11px;
}

/* ========== Inputs ========== */
QLineEdit,
QComboBox,
QDoubleSpinBox {
    min-height: 24px;
    padding: 6px 10px;
    border: 1px solid #DDE3EA;
    border-radius: 7px;
    background-color: #FBFCFD;
    color: #1F2937;
    selection-background-color: #CFE1EA;
}

QLineEdit[readOnly="true"] {
    background-color: #F7F8FA;
    color: #6B7280;
}

QLineEdit:hover,
QComboBox:hover,
QDoubleSpinBox:hover {
    border-color: #C5D0DA;
}

QLineEdit:focus,
QComboBox:focus,
QDoubleSpinBox:focus {
    border-color: #7EA0B8;
    background-color: #FFFFFF;
}

QComboBox::drop-down {
    width: 26px;
    border-left: 1px solid #E5E7EB;
    background-color: #F2F4F7;
    border-top-right-radius: 7px;
    border-bottom-right-radius: 7px;
}

QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #DDE3EA;
    selection-background-color: #E6F0F4;
    selection-color: #2B4C5F;
    padding: 4px;
}

QDoubleSpinBox::up-button,
QDoubleSpinBox::down-button {
    width: 16px;
    border: none;
    background-color: #EEF2F5;
}

QRadioButton,
QCheckBox {
    color: #4B5563;
    font-size: 12px;
    spacing: 8px;
}

QRadioButton::indicator,
QCheckBox::indicator {
    width: 14px;
    height: 14px;
}

QRadioButton::indicator:checked,
QCheckBox::indicator:checked {
    background-color: #4F8A8B;
    border: 1px solid #4F8A8B;
}

QRadioButton::indicator:unchecked,
QCheckBox::indicator:unchecked {
    background-color: #FFFFFF;
    border: 1px solid #C9D6E2;
}

QSlider::groove:horizontal {
    height: 5px;
    border-radius: 2px;
    background-color: #DDE3EA;
}

QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background-color: #5B7C99;
}

/* ========== Buttons ========== */
QPushButton {
    min-height: 28px;
    padding: 7px 14px;
    border-radius: 7px;
    border: 1px solid #C9D6E2;
    background-color: #FFFFFF;
    color: #374151;
    font-size: 12px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #F2F6F8;
    border-color: #9DB4C5;
    color: #2B4C5F;
}

QPushButton:pressed {
    background-color: #E6EDF3;
}

QPushButton:disabled {
    background-color: #EEF2F5;
    border-color: #DDE3EA;
    color: #9AA5B1;
}

QPushButton#btnGenerate {
    border: none;
    background-color: #4F8A8B;
    color: #FFFFFF;
    min-height: 34px;
}

QPushButton#btnGenerate:hover {
    background-color: #467E80;
}

QPushButton#btnDetect {
    border: none;
    background-color: #5D9C8A;
    color: #FFFFFF;
    min-height: 34px;
}

QPushButton#btnDetect:hover {
    background-color: #528E7E;
}

QPushButton#btnOpenDir {
    color: #4F6F8F;
    background-color: #F7FAFB;
}

QPushButton#btnOpenDir:hover {
    background-color: #EEF5F7;
}

/* ========== Image, Logs, Progress ========== */
QLabel#imageCanvas {
    background-color: #F8FAFC;
    color: #7A8794;
    border: none;
    font-size: 13px;
}

QTextEdit {
    border: 1px solid #E5E7EB;
    border-radius: 7px;
    background-color: #FBFCFD;
    color: #374151;
    padding: 6px;
    font-size: 11px;
    selection-background-color: #CFE1EA;
}

QTextEdit#logText {
    background-color: #FBFCFD;
    color: #5D6673;
}

QTextEdit#detailText,
QTextEdit#classText {
    background-color: #F8FAFC;
    color: #2F4050;
}

QScrollBar:vertical {
    background-color: #F2F4F7;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: #C5D0DA;
    border-radius: 4px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background-color: #9AAABD;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0px;
}

QProgressBar {
    border: 1px solid #DDE3EA;
    border-radius: 6px;
    background-color: #F2F4F7;
    color: #4B5563;
    text-align: center;
    min-height: 12px;
}

QProgressBar::chunk {
    border-radius: 5px;
    background-color: #5B7C99;
}

/* ========== Status Bar ========== */
QStatusBar {
    background-color: #FFFFFF;
    color: #6B7280;
    border-top: 1px solid #DDE3EA;
    padding: 4px 12px;
}

QLabel#statusText {
    color: #6B7280;
    font-size: 11px;
}
"""
