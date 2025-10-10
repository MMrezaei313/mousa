# استفاده از image رسمی پایتون
FROM python:3.11-slim

# تنظیم متغیرهای محیطی
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# ایجاد دایرکتوری برنامه
WORKDIR $APP_HOME

# نصب وابستگی‌های سیستم
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# کپی فایل requirements
COPY requirements.txt .

# نصب وابستگی‌های پایتون
RUN pip install --no-cache-dir -r requirements.txt

# کپی کد برنامه
COPY . .

# ایجاد کاربر غیر root
RUN useradd --create-home --shell /bin/bash mousa
USER mousa

# پورت اکسپوز
EXPOSE 5000

# کامند اجرا
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "mousatrade.web.app:app"]
