# 🚀 Streamlit Cloud 部署指南

## 📋 准备工作

### 1. 确保代码已推送到 GitHub
```powershell
git status
git add .
git commit -m "feat: Add Streamlit Cloud deployment support"
git push
```

### 2. 注册 Streamlit Cloud
1. 访问：https://share.streamlit.io/
2. 用你的 GitHub 账号登录
3. 授权 Streamlit 访问你的仓库

---

## 🔧 部署步骤

### 步骤 1：创建新应用

1. 点击 **"New app"**
2. 选择你的仓库：`Jack-Fang-618/AI-Screening-Tool-for-Literature-Review`
3. 选择分支：`main`
4. Main file path: `streamlit_app.py`
5. 点击 **"Deploy!"**

### 步骤 2：配置环境变量（重要！）

在部署页面：

1. 点击 **"Advanced settings"** 或部署后点击 **"Manage app" → "Settings"**
2. 找到 **"Secrets"** 部分
3. 添加你的 API Key：

```toml
XAI_API_KEY = "your-xai-api-key-here"
```

4. 保存设置

### 步骤 3：等待部署完成

- 首次部署需要 3-5 分钟
- 安装所有依赖（从 `requirements.txt`）
- 启动 FastAPI 后端 + Streamlit 前端

### 步骤 4：获取分享链接

部署成功后，你会得到一个链接：
```
https://your-app-name.streamlit.app
```

---

## 🔒 设置私有访问（推荐）

### 添加密码保护：

在 Streamlit Cloud 控制台：

1. 进入你的应用设置
2. 找到 **"Sharing"** 部分
3. 选择 **"Private"**
4. 添加允许访问的 email 地址（你的合作者）

或者在代码中添加简单密码验证：

```python
# 在 streamlit_app.py 开头添加
import streamlit as st

def check_password():
    """简单的密码保护"""
    def password_entered():
        if st.session_state["password"] == "your-secret-password":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "请输入密码", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "请输入密码", type="password", on_change=password_entered, key="password"
        )
        st.error("密码错误")
        return False
    else:
        return True

if not check_password():
    st.stop()

# 继续正常的应用代码...
```

---

## 📊 性能优化建议

### 1. 冷启动优化

Streamlit Cloud 在无访问时会休眠，重新启动需要 5-10 秒。解决方法：

- 使用 UptimeRobot（免费）定期 ping 你的网址保持唤醒
- 或者告诉用户首次访问需要等待

### 2. 数据库位置

Streamlit Cloud 的文件系统是临时的，重启后会丢失。建议：

**选项 A：使用外部数据库（推荐）**
```python
# 使用 Supabase（免费 PostgreSQL）
DATABASE_URL = st.secrets["DATABASE_URL"]
```

**选项 B：使用 Streamlit Session State**
```python
# 数据存在内存里，用户刷新页面会丢失
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
```

**选项 C：使用 GitHub 作为存储**
```python
# 将结果保存到 GitHub Gist
# 适合小数据量
```

### 3. 资源限制

Streamlit Community Cloud 免费层限制：
- **CPU**: 1 核心（共享）
- **RAM**: 1GB
- **存储**: 临时，重启后清空
- **并发**: 支持多用户，但性能会下降

如果需要更多资源，考虑：
- Streamlit Cloud 付费版（$20/月起）
- 或者用方案 2（前后端分离部署）

---

## 🐛 常见问题

### 问题 1：Backend 启动失败

**错误信息**：`Address already in use`

**解决**：
```python
# 在 streamlit_app.py 里改成随机端口
import socket
def get_free_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]

port = get_free_port()
uvicorn.run(app, host="0.0.0.0", port=port)
```

### 问题 2：API 调用超时

**原因**：Backend 还没启动完成

**解决**：在 `streamlit_app.py` 里增加等待时间：
```python
time.sleep(5)  # 从 3 秒改成 5 秒
```

### 问题 3：数据库文件丢失

**原因**：Streamlit Cloud 重启后文件系统清空

**解决**：
- 使用外部数据库（Supabase/PlanetScale）
- 或者让用户每次上传数据

### 问题 4：环境变量读取失败

**解决**：
```python
import os
import streamlit as st

# 优先从 Streamlit secrets 读取
if "XAI_API_KEY" in st.secrets:
    api_key = st.secrets["XAI_API_KEY"]
else:
    # 本地开发用 .env
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("XAI_API_KEY")
```

---

## 📱 分享给朋友

部署成功后，直接把链接发给朋友：

```
https://ai-screening-tool.streamlit.app
```

他们：
1. ✅ 打开链接就能用（如果是 Public）
2. ✅ 输入密码/邮箱验证后使用（如果是 Private）
3. ✅ 不需要安装任何软件
4. ✅ 不需要配置环境
5. ✅ 任何设备都能访问（电脑、手机、平板）

---

## 💰 成本

**Streamlit Community Cloud（免费层）**：
- ✅ 完全免费
- ✅ 无限制部署数量
- ✅ 私有链接 + 密码保护
- ⚠️ 资源有限（1 CPU, 1GB RAM）
- ⚠️ 冷启动慢

**升级到付费版（$20/月）**：
- ✅ 更多 CPU 和 RAM
- ✅ 更快启动速度
- ✅ 优先支持
- ✅ 自定义域名

---

## 🔄 更新应用

当你修改代码后：

```powershell
# 本地修改代码
git add .
git commit -m "fix: Update screening logic"
git push
```

Streamlit Cloud 会自动：
1. 检测到 GitHub 更新
2. 自动重新部署
3. 2-3 分钟后生效

或者在 Streamlit Cloud 控制台手动点击 **"Reboot app"**。

---

## 📞 需要帮助？

遇到问题：
1. 查看 Streamlit Cloud 的日志（控制台右下角）
2. 检查 GitHub Actions 是否有错误
3. 在 Streamlit Community 论坛提问
4. 或者联系我：wennbo@hku.hk

---

**准备好了吗？开始部署吧！** 🚀
