from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
from dotenv import load_dotenv
from nltk.chat.util import Chat, reflections
import asyncio
import joblib

from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")





async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello sir/mam to kisaan-saathi portal...😀")
    await asyncio.sleep(2)
    await update.message.reply_text("As kisaan is working for feeding us, we are taking a small step to help kisaan")
    await asyncio.sleep(2)
    await update.message.reply_text('This Portal is for detecting the diseases in plants leaves, so just click picture, send it, and know the diseases...')


async def handle_photo(update:Update,context:ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1] 
    photo_file = await photo.get_file()
    photo_path = f'photos/{photo_file.file_id}.jpg'  # Define a file path and name
    
    # Download and save the photo
    await photo_file.download_to_drive(photo_path)
    await update.message.reply_text("Photo saved successfully!")


class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus']

async def Leaf_diseases_prediction(update:Update,context:ContextTypes.DEFAULT_TYPE):
    cnn_model = joblib.load(r"C:\Users\Public\Documents\Python_project\plant_diseases_detector.pkl")

    folder_path = r"C:\Users\svish\OneDrive\Desktop\plant_diseases_detection\photos"
    count = 0
    image_path = ''
    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path,image)
        count+=1
        if(count==1):
            break
    
    await update.message.reply_text("Analyzing...")
    
    image_shape=(64,64,3)
    image = load_img(image_path,target_size=image_shape[:2])
    image_array = img_to_array(image)  # Convert to array
    image_array = image_array / 255  # Normalize (if your model was trained with normalized images)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = cnn_model.predict(image_array)
    predicted_class = np.argmax(predictions,axis=1)
    predicted_class_index = np.argmax(predictions,axis=1)[0]
    predicted_probability = predictions[0][predicted_class]*100
    await asyncio.sleep(5)
    await update.message.reply_text(f"probability: {predicted_probability}")
    # if(predicted_probability >= 75):
    #     await update.message.reply_text(f"diseases detected : {class_names[predicted_class_index]}")
    # else:

    #     await update.message.reply_text(f"provide an appropriate and clear image, it seems pic other than leaf of plants ")

    
    
async def delete_images(update:Update,context:ContextTypes.DEFAULT_TYPE):
    folder = r"C:\Users\svish\OneDrive\Desktop\plant_diseases_detection\photos"
    for img in os.listdir(folder):
        img_path = os.path.join(folder,img)
        os.remove(img_path)

    await update.message.reply_text("successfully deleted")







def main():
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO,handle_photo))
    app.add_handler(CommandHandler("detect", Leaf_diseases_prediction))
    app.add_handler(CommandHandler("delete", delete_images))
    
    app.run_polling()

if __name__ == '__main__':
    main()
