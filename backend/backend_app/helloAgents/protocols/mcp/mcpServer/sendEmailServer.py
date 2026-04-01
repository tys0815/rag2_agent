import sys
import os

#定位到当前文件所在目录
CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)

# 向上跳 4 级，到达 backend_app（项目根目录）
# 层级：mcpServer → mcp → protocols → helloAgents → backend_app
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))

# 加入 Python 路径
sys.path.insert(0, PROJECT_ROOT)

import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from helloAgents.protocols import MCPServer
import json
from typing import Annotated
from pydantic import Field

send_qq_email_server = MCPServer(name="qq_email_server", description="发送QQ邮件的MCP服务器")

def send_qq_email(
    to_user: Annotated[str, Field(description="收件人邮箱地址，例如 123456@qq.com")],
    title: Annotated[str, Field(description="邮件标题")],
    content: Annotated[str, Field(description="邮件正文内容")]
) -> str:
    """发送 QQ 邮件"""
    # ———— 必须改成你自己的 ————
    from_user = "2981701991@qq.com"
    password = "gnflcacdegkddhcb"  # 不是密码！
    # ————————————————————————

    try:
        # 构建邮件
        msg = MIMEText(content, "plain", "utf-8")
        msg["From"] = formataddr(["发件人昵称", from_user])
        msg["To"] = formataddr(["收件人", to_user])
        msg["Subject"] = title

        # QQ 邮箱服务器
        server = smtplib.SMTP_SSL("smtp.qq.com", 465)
        server.login(from_user, password)
        server.sendmail(from_user, [to_user], msg.as_string())
        server.quit()
        return "发送成功！"

    except Exception as e:
        return f"发送失败：{e}"

def get_server_info() -> str:
    """获取服务器信息"""
    info = {
        "name": "sendQQEmail MCP Server",
        "version": "1.0.0",
        "tools": ["send_qq_email", "get_server_info"]
    }
    return json.dumps(info, ensure_ascii=False, indent=2)

send_qq_email_server.add_tool(send_qq_email)
send_qq_email_server.add_tool(get_server_info)

# 直接调用
if __name__ == "__main__":
    # send_qq_email(
    #     to_user="2124965021@qq.com",
    #     title="测试邮件",
    #     content="这是用 Python 发送的 QQ 邮件！"
    # )
    send_qq_email_server.run()