# THA4 模型使用指南 (新版本)

## 文件结构

THA4 模型现在支持直接放在 `data/images/` 目录中，与 THA3 的 PNG 图片一起管理。

### 所需文件

对于名为 `mycharacter` 的角色，需要在 `data/images/` 中放置以下文件：

```
data/images/
├── mycharacter.yaml                 # 配置文件
├── mycharacter.png                  # 角色源图片 (512x512 RGBA)
├── mycharacter_face_morpher.pt      # 面部变形模型权重
└── mycharacter_body_morpher.pt      # 身体变形模型权重
```

### YAML 配置格式

YAML 文件 (`mycharacter.yaml`) 应包含相对路径指向其他文件：

```yaml
character_image_file_name: mycharacter.png
face_morpher_file_name: mycharacter_face_morpher.pt
body_morpher_file_name: mycharacter_body_morpher.pt
```

**重要说明：**
- YAML 中的所有路径都是相对于 YAML 文件所在位置的相对路径
- 由于所有文件都在 `data/images/` 中，只需使用文件名即可
- 角色图片必须是 512x512 像素的 RGBA 格式
- 模型文件 (`.pt`) 是从 THA4 训练得到的 PyTorch 权重

## 在 EasyVtuber 中使用

### 自动检测

1. 将所有 4 个文件放入 `data/images/` 目录
2. 打开启动器 (`02A.启动器.bat`)
3. 在 Character 下拉菜单中选择 `.yaml` 文件（例如 `mycharacter`）
4. 程序将自动：
   - 检测这是 THA4 模型
   - 切换到 THA4 推理模式
   - 从 YAML 配置加载角色图片和模型

### 模型选择锁定

当你选择 YAML 文件时：
- 模型**自动锁定到 THA4**
- 使用 YAML 中的角色图片（启动器中的 Character 选择仅用于选择 YAML）
- 手动的模型选择将被覆盖

## 示例设置

假设你为名为 "alice" 的角色训练了 THA4 模型：

1. 将这些文件复制到 `data/images/`：
   ```
   alice.yaml
   alice.png
   alice_face_morpher.pt
   alice_body_morpher.pt
   ```

2. 创建 `alice.yaml`，内容为：
   ```yaml
   character_image_file_name: alice.png
   face_morpher_file_name: alice_face_morpher.pt
   body_morpher_file_name: alice_body_morpher.pt
   ```

3. 在启动器中，从 Character 下拉菜单选择 "alice"
4. 启动 - 程序将检测到 YAML 并自动使用 THA4！

## 与 THA3 的区别

| 特性 | THA3 | THA4 |
|------|------|------|
| 角色选择 | data/images/ 中的任意 PNG | 选择 YAML 文件 |
| 模型文件 | data/models/tha3/ 中的共享模型 | 每个角色独立的 .pt 文件 |
| 图片加载 | 运行时选择 | 与训练模型绑定 |
| 模型类型 | Standard/Separate, Half/Full | 单一类型 (float32) |
| 可互换性 | 可使用任意图片 | 图片与训练模型绑定 |

## 文件命名建议

为了保持清晰，建议使用一致的命名规则：

```
<角色名>.yaml           # 配置文件
<角色名>.png            # 源图片
<角色名>_face_morpher.pt   # 面部模型
<角色名>_body_morpher.pt   # 身体模型
```

## 故障排除

**错误："THA4 character model YAML not found"**
- 检查 `.yaml` 文件是否存在于 `data/images/` 中
- 验证文件名是否与你选择的一致

**错误："No such file or directory" 找不到 PNG 或 PT 文件**
- 检查 YAML 中的路径是否正确
- 确保所有 4 个文件都在同一个 `data/images/` 目录中
- 验证文件名完全匹配（区分大小写）

**黑屏 / 无输出**
- 验证角色 PNG 是 512x512 RGBA 格式
- 检查 .pt 模型文件是否来自正确训练的 THA4 模型
- 确保模型文件与用于训练的角色图片匹配

## 旧文件迁移

如果你之前将文件放在 `data/models/tha4/` 中：

1. 将所有文件移动到 `data/images/`：
   ```
   mv data/models/tha4/character_model.yaml data/images/mycharacter.yaml
   mv data/models/tha4/character.png data/images/mycharacter.png
   mv data/models/tha4/face_morpher.pt data/images/mycharacter_face_morpher.pt
   mv data/models/tha4/body_morpher.pt data/images/mycharacter_body_morpher.pt
   ```

2. 更新 YAML 文件中的路径为新的文件名

3. 在启动器中选择新的 YAML 文件名
