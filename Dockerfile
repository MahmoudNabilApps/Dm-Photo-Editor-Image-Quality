FROM python:3.10-slim

# تعيين متغيرات البيئة
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# تثبيت المكتبات الأساسية للنظام
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*
# إنشاء مجلد العمل
WORKDIR /app

# نسخ ملف المتطلبات
COPY requirements.txt .

# تثبيت المكتبات Python
RUN pip install --no-cache-dir -r requirements.txt

# تنزيل نموذج RealESRGAN (لتجنب التنزيل عند كل تشغيل)
RUN wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O /app/RealESRGAN_x4plus.pth

# نسخ كود التطبيق
COPY app.py .

# فتح المنفذ 7860 (المنفذ الافتراضي لـ Hugging Face Spaces)
EXPOSE 7860

# تشغيل التطبيق
CMD ["python", "app.py"]
