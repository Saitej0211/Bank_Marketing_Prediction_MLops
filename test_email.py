import smtplib

smtp_host = "smtp.gmail.com"
smtp_port = 587

try:
    server = smtplib.SMTP(host=smtp_host, port=smtp_port)
    server.starttls()  # Upgrade to a secure connection
    print("Connection to SMTP server successful")
    server.quit()
except Exception as e:
    print(f"Failed to connect: {e}")
