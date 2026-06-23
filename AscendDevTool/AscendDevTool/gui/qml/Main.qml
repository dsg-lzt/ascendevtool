import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import Qt.labs.platform 1.1

ApplicationWindow {
    id: win
    width: 980
    height: 680
    visible: true
    title: "Pytorch VLA 昇腾迁移工具 - Scanner GUI"

    // 让表格列宽随窗口大小变化（保留最小/最大宽度，避免过窄/过宽）
    property int countColWidth: Math.max(90, Math.min(220, Math.round(win.width * 0.22)))

    ListModel {
        id: unsupportedTableModel
    }

    function refreshUnsupportedTable() {
        unsupportedTableModel.clear()
        var raw = backend.unsupportedOpsJson
        if (!raw || raw.length === 0)
            raw = "[]"
        console.log("refreshUnsupportedTable: json_len=", raw.length)
        try {
            var arr = JSON.parse(raw)
            console.log("refreshUnsupportedTable: parsed_len=", (arr && arr.length !== undefined) ? arr.length : "n/a")
            if (!arr)
                return
            for (var i = 0; i < arr.length; i++) {
                var it = arr[i]
                if (!it)
                    continue
                var opv = (it.op !== undefined) ? it.op : it["op"]
                var cv = (it.count !== undefined) ? it.count : it["count"]
                unsupportedTableModel.append({
                    opText: (opv === undefined || opv === null) ? "" : String(opv),
                    cnt: (cv === undefined || cv === null || cv === "") ? 0 : Number(cv)
                })
            }
            console.log("refreshUnsupportedTable: model_count=", unsupportedTableModel.count)
            if (unsupportedTableModel.count > 0) {
                var first = unsupportedTableModel.get(0)
                console.log("refreshUnsupportedTable: first_row=", first.opText, first.cnt)
            }
        } catch (e) {
            console.log("refreshUnsupportedTable: JSON.parse failed:", e)
        }
    }

    Connections {
        target: backend
        function onUnsupportedOpsJsonChanged() { refreshUnsupportedTable() }
        function onSummaryChanged() { refreshUnsupportedTable() }
    }

    Component.onCompleted: refreshUnsupportedTable()

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        GroupBox {
            title: "路径选择"
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent
                spacing: 10

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Label { text: "待迁移模型目录"; Layout.preferredWidth: 140 }
                    TextField {
                        id: sourceField
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        readOnly: true
                        clip: true
                        selectByMouse: true
                        text: backend.sourceDir
                        placeholderText: "请选择包含待迁移模型代码/脚本的目录"
                    }
                    Button {
                        text: "浏览..."
                        onClicked: sourceDialog.open()
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Label { text: "迁移目标目录"; Layout.preferredWidth: 140 }
                    TextField {
                        id: targetField
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        readOnly: true
                        clip: true
                        selectByMouse: true
                        text: backend.targetDir
                        placeholderText: "请选择迁移输出目录（本期仅保存，不参与扫描）"
                    }
                    Button {
                        text: "浏览..."
                        onClicked: targetDialog.open()
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Label { text: "PyTorch 版本"; Layout.preferredWidth: 140 }
                    ComboBox {
                        id: torchVersionBox
                        Layout.preferredWidth: 180
                        model: backend.torchVersionOptions

                        Component.onCompleted: {
                            var idx = find(backend.torchVersion)
                            if (idx >= 0)
                                currentIndex = idx
                        }

                        onActivated: {
                            backend.torchVersion = currentText
                        }

                        Connections {
                            target: backend
                            function onTorchVersionChanged() {
                                var idx = torchVersionBox.find(backend.torchVersion)
                                if (idx >= 0 && torchVersionBox.currentIndex !== idx)
                                    torchVersionBox.currentIndex = idx
                            }
                        }
                    }
                    Label {
                        Layout.fillWidth: true
                        color: "#666666"
                        text: "默认：使用已安装 torch 版本；若未安装则使用 2.6.0"
                        elide: Text.ElideRight
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Label { text: "已扫描结果路径"; Layout.preferredWidth: 140 }
                    TextField {
                        id: reportField
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        readOnly: true
                        clip: true
                        selectByMouse: true
                        text: backend.reportPath
                        placeholderText: "可选：选择 scanner/reports 下任意历史结果目录"
                    }
                    Button {
                        text: "浏览..."
                        onClicked: reportDialog.open()
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Button {
                        text: backend.busy ? "扫描中..." : "扫描模型"
                        enabled: !backend.busy
                        onClicked: backend.scanModels()
                    }
                    Button {
                        text: "加载已扫描结果"
                        enabled: !backend.busy
                        onClicked: backend.loadScannedResult()
                    }
                    BusyIndicator {
                        running: backend.busy
                        visible: backend.busy
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: backend.totalUniqueOps > 0 ? ("不支持算子种类: " + backend.totalUniqueOps + "，出现次数: " + backend.totalOccurrences) : ""
                        horizontalAlignment: Text.AlignRight
                        Layout.alignment: Qt.AlignRight
                    }
                }

                ScrollView {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 110
                    Layout.maximumHeight: 180

                    TextArea {
                        readOnly: true
                        selectByMouse: true
                        wrapMode: TextEdit.Wrap
                        text: backend.status
                        color: "#666666"
                    }
                }
            }
        }

        GroupBox {
            title: "不支持算子统计（按出现次数降序）"
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                anchors.fill: parent
                spacing: 1

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Label { text: "OP"; font.bold: true; Layout.fillWidth: true }
                    Label { text: "Count"; font.bold: true; Layout.preferredWidth: win.countColWidth; horizontalAlignment: Text.AlignRight }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 1
                    color: "#dddddd"
                }

                ListView {
                    id: list
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 120
                    clip: true
                    model: unsupportedTableModel

                    delegate: Item {
                        width: list.width
                        height: 34

                        Component.onCompleted: {
                            console.log("delegate:", index, opText, cnt)
                        }

                        RowLayout {
                            anchors.fill: parent
                            spacing: 8

                            Label {
                                text: opText
                                elide: Text.ElideRight
                                Layout.fillWidth: true
                            }
                            Label {
                                text: cnt
                                Layout.preferredWidth: win.countColWidth
                                horizontalAlignment: Text.AlignRight
                            }
                        }

                        Rectangle {
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.bottom: parent.bottom
                            height: 1
                            color: "#f0f0f0"
                        }
                    }

                    ScrollBar.vertical: ScrollBar {}
                }
            }
        }
    }

    FolderDialog {
        id: sourceDialog
        title: "选择待迁移模型目录"
        onAccepted: {
            // folder is url (file:///...)
            backend.sourceDir = sourceDialog.folder.toString().replace(/^file:\/\//, "")
     
        }
    }

    FolderDialog {
        id: targetDialog
        title: "选择迁移目标目录"
        onAccepted: {
            backend.targetDir = targetDialog.folder.toString().replace(/^file:\/\//, "")
        }
    }

    FolderDialog {
        id: reportDialog
        title: "选择已扫描结果路径"
        onAccepted: {
            backend.reportPath = reportDialog.folder.toString().replace(/^file:\/\//, "")
        }
    }
}
