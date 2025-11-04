# VSCode C++ 开发指南

本指南说明如何在VSCode中编译和调试C++项目。

## 文件说明

### 1. `settings.json` - 编辑器设置
配置VSCode的基本行为，包括：
- **CMake配置**: 告诉VSCode如何使用CMake编译项目
- **IntelliSense**: 代码补全和错误检查
- **调试器设置**: 调试时的显示选项

### 2. `tasks.json` - 编译任务
定义了三个任务：
- **CMake: Build** - 编译项目（快捷键: `Ctrl+Shift+B`）
- **CMake: Clean** - 清理编译文件
- **Run: demo_test** - 运行编译后的程序

### 3. `launch.json` - 调试配置
定义了调试器的行为：
- 使用GDB作为调试器
- 自动在调试前编译项目
- 设置工作目录为build/bin（确保程序能访问Data目录）

## 使用流程

### 第一次使用

1. **打开项目**
   - 在VSCode中打开此项目文件夹
   - VSCode会自动检测CMakeLists.txt并配置CMake

2. **编译项目**
   - 按 `Ctrl+Shift+B` 执行默认编译任务
   - 或在命令面板(Ctrl+Shift+P)中选择 "Tasks: Run Build Task"
   - 编译输出会显示在下方的终端中

3. **调试程序**
   - 在代码中点击行号左侧设置断点（红点）
   - 按 `F5` 启动调试
   - 程序会在断点处停止，你可以：
     - 查看变量值（左侧Variables面板）
     - 逐行执行代码（F10）
     - 进入函数内部（F11）
     - 继续执行到下一个断点（F5）

### 常用快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+Shift+B` | 编译项目 |
| `F5` | 启动/继续调试 |
| `F10` | 逐行执行（不进入函数） |
| `F11` | 进入函数 |
| `Shift+F11` | 退出函数 |
| `Ctrl+Shift+D` | 打开调试面板 |

## 调试技巧

### 查看变量值
1. 在调试时，左侧"Variables"面板显示当前作用域的所有变量
2. 可以在"Watch"面板中添加表达式来监视特定变量
3. 将鼠标悬停在代码中的变量上，会显示其当前值

### 条件断点
1. 右键点击断点（红点）
2. 选择"Edit Breakpoint"
3. 输入条件表达式，例如 `i > 100`
4. 只有当条件为真时，程序才会在此断点停止

### 调试控制台
- 在调试时，可以在下方的"Debug Console"中执行GDB命令
- 例如：`print variable_name` 查看变量值

## 常见问题

### 编译失败
- 检查CMakeLists.txt是否正确
- 确保已安装OpenCV库：`pkg-config --modversion opencv4`
- 查看编译输出中的错误信息

### 调试时看不到变量值
- 确保编译时使用了调试符号（-g标志）
- CMakeLists.txt中应该有 `set(CMAKE_BUILD_TYPE Debug)`

### 程序找不到Data目录
- 检查launch.json中的"cwd"是否设置为 `${workspaceFolder}/build/bin`
- 确保Data目录已复制到build/bin目录中

## 进阶配置

如果需要更复杂的功能，可以修改这些文件：
- 添加多个调试配置（例如不同的编译目标）
- 自定义编译选项
- 添加预启动任务（如自动清理旧编译文件）

详见VSCode官方文档：https://code.visualstudio.com/docs/cpp/config-linux