import os
import cv2
import base64
import numpy as np
import torch
from flask import Flask, request, jsonify
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# تهيئة Flask
app = Flask(__name__)

# تعطيل استخدام GPU وفرض CPU
torch.set_num_threads(4)  # استخدام 4 threads للمعالجة المتوازية
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("="*50)
print("🚀 بدء تحميل نموذج Real-ESRGAN...")
print("="*50)

# إعداد النموذج
model_path = '/app/RealESRGAN_x4plus.pth'

# التحقق من وجود الملف
if not os.path.exists(model_path):
    print(f"❌ خطأ: النموذج غير موجود في {model_path}")
    raise FileNotFoundError(f"Model file not found: {model_path}")

# تحميل نموذج RealESRGAN
model = RRDBNet(
    num_in_ch=3, 
    num_out_ch=3, 
    num_feat=64, 
    num_block=23, 
    num_grow_ch=32, 
    scale=4
)

# إنشاء upscaler مُحسّن للـ CPU
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=256,  # ✅ استخدام tiles صغيرة لتقليل استخدام الذاكرة
    tile_pad=10,
    pre_pad=0,
    half=False,  # ✅ لا نستخدم half precision مع CPU
    device='cpu'  # ✅ فرض استخدام CPU
)

print("✅ تم تحميل النموذج بنجاح!")
print(f"📊 الجهاز المستخدم: CPU")
print(f"🧵 عدد الـ Threads: {torch.get_num_threads()}")
print("="*50)


@app.route('/')
def home():
    """الصفحة الرئيسية"""
    return '''
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Real-ESRGAN API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 { color: #2c3e50; }
            .endpoint {
                background: #ecf0f1;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }
            code {
                background: #2c3e50;
                color: #ecf0f1;
                padding: 2px 6px;
                border-radius: 3px;
            }
            .status { color: #27ae60; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 Real-ESRGAN Image Upscaling API</h1>
            <p class="status">✅ API يعمل بشكل صحيح - 24/7 مجاني على Hugging Face!</p>
            
            <h2>📌 Endpoints المتاحة:</h2>
            
            <div class="endpoint">
                <strong>GET /health</strong>
                <p>فحص حالة السيرفر والتأكد من جاهزيته</p>
            </div>
            
            <div class="endpoint">
                <strong>POST /upscale</strong>
                <p>رفع جودة الصورة بمقدار 4x</p>
                <pre><code>{
  "image": "base64_encoded_image_here"
}</code></pre>
            </div>
            
            <h2>💡 ملاحظات مهمة:</h2>
            <ul>
                <li>API يعمل على CPU (قد يستغرق وقتاً أطول من GPU)</li>
                <li>الصور الكبيرة قد تستغرق 10-30 ثانية للمعالجة</li>
                <li>يُنصح بإرسال صور بحجم أقصى 1000x1000 بكسل</li>
            </ul>
        </div>
    </body>
    </html>
    '''


@app.route('/health', methods=['GET'])
def health():
    """فحص حالة API"""
    return jsonify({
        'status': 'healthy',
        'message': '✅ API يعمل بشكل صحيح',
        'model': 'RealESRGAN_x4plus',
        'device': 'CPU',
        'uptime': '24/7 مضمون',
        'platform': 'Hugging Face Spaces'
    }), 200


@app.route('/upscale', methods=['POST'])
def upscale_image():
    """
    رفع جودة الصورة
    يستقبل صورة بصيغة base64 ويعيد الصورة المحسنة
    """
    try:
        # التحقق من وجود البيانات
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type يجب أن يكون application/json'
            }), 400
        
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'لم يتم إرسال صورة (مطلوب حقل "image")'
            }), 400
        
        # فك تشفير base64
        image_data = data['image']
        
        # إزالة header إذا وجد (data:image/png;base64,...)
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # تحويل base64 إلى صورة
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'خطأ في فك تشفير base64: {str(e)}'
            }), 400
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'فشل في قراءة الصورة - تأكد من صحة الصورة'
            }), 400
        
        original_height, original_width = img.shape[:2]
        print(f"📸 تم استقبال صورة: {original_width}x{original_height}")
        
        # ✅ تحذير إذا كانت الصورة كبيرة جداً
        if original_width > 1500 or original_height > 1500:
            return jsonify({
                'success': False,
                'error': 'الصورة كبيرة جداً. الحد الأقصى المسموح: 1500x1500 بكسل',
                'current_size': f"{original_width}x{original_height}"
            }), 400
        
        # رفع جودة الصورة
        print("⚙️ جاري معالجة الصورة... (قد يستغرق دقيقة)")
        output, _ = upsampler.enhance(img, outscale=4)
        
        upscaled_height, upscaled_width = output.shape[:2]
        print(f"✨ تم تحسين الصورة إلى: {upscaled_width}x{upscaled_height}")
        
        # تحويل الصورة المحسنة إلى base64
        _, buffer = cv2.imencode('.png', output, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        output_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # إرجاع النتيجة
        return jsonify({
            'success': True,
            'message': 'تم تحسين الصورة بنجاح',
            'original_size': f"{original_width}x{original_height}",
            'upscaled_size': f"{upscaled_width}x{upscaled_height}",
            'upscaled_image': f"data:image/png;base64,{output_base64}"
        }), 200
        
    except Exception as e:
        print(f"❌ خطأ: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'خطأ في المعالجة: {str(e)}'
        }), 500


if __name__ == '__main__':
    # Hugging Face Spaces يستخدم المنفذ 7860
    port = int(os.environ.get('PORT', 7860))
    
    print("\n" + "="*50)
    print(f"🚀 تشغيل السيرفر على المنفذ {port}")
    print("="*50 + "\n")
    
    # تشغيل Flask
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True  # ✅ دعم multiple requests
    )
