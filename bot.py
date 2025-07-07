import logging
import os
import re
import asyncio
import aiosqlite
from dotenv import load_dotenv
import telegram
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    ConversationHandler
)
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Загрузка переменных окружения
load_dotenv()

# Настройки из .env
TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID"))
DB_NAME = os.getenv("DB_NAME", "acm_med_bot.db")
AI_THRESHOLD = float(os.getenv("AI_THRESHOLD", 0.5))
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Состояния для ConversationHandler
ADMIN_LOGIN, ADMIN_MODE = range(2)

# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Глобальные переменные для кеширования ИИ-модели
vectorizer = TfidfVectorizer()
vectorizer_fitted = False
qa_vectors = None
qa_cache = []

# Глобальное соединение с БД
db_connection = None
db_lock = asyncio.Lock()

async def get_db():
    global db_connection
    if db_connection is None:
        db_connection = await aiosqlite.connect(DB_NAME, timeout=30)
        logger.info(f"Соединение с {DB_NAME} установлено")
    return db_connection

async def close_db():
    global db_connection
    if db_connection:
        await db_connection.close()
        logger.info("Соединение с БД закрыто")

async def column_exists(conn, table_name, column_name):
    """Проверяет существование столбца в таблице."""
    cursor = await conn.execute(f"PRAGMA table_info({table_name})")
    columns = await cursor.fetchall()
    for col in columns:
        if col[1] == column_name:
            return True
    return False

# Инициализация базы данных
async def init_db():
    async with db_lock:
        conn = await get_db()
        
        # Создание таблиц
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT UNIQUE,
                answer TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                answered_by TEXT DEFAULT 'system',
                last_used TIMESTAMP,
                use_count INTEGER DEFAULT 0,
                user_id INTEGER
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                question TEXT,
                answer TEXT,
                feedback_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        ''')
        
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE,
                username TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Проверяем и добавляем отсутствующие столбцы в qa
        columns_to_check = [
            ('use_count', 'INTEGER DEFAULT 0'),
            ('last_used', 'TIMESTAMP'),
            ('answered_by', 'TEXT DEFAULT \'system\''),
            ('user_id', 'INTEGER')
        ]
        for column, col_type in columns_to_check:
            if not await column_exists(conn, 'qa', column):
                logger.info(f"Добавляем столбец {column} в таблицу qa")
                await conn.execute(f'ALTER TABLE qa ADD COLUMN {column} {col_type}')
        
        # Добавляем основного администратора
        try:
            await conn.execute('''
                INSERT OR IGNORE INTO admins (user_id) VALUES (?)
            ''', (ADMIN_ID,))
        except aiosqlite.IntegrityError:
            pass
        
        # Создаем индексы для оптимизации
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_qa_question ON qa(question)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_qa_answer ON qa(answer)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_qa_use_count ON qa(use_count)')
        
        await conn.commit()
    logger.info(f"База данных {DB_NAME} инициализирована")

# Проверка прав администратора
async def is_admin(user_id: int) -> bool:
    async with db_lock:
        conn = await get_db()
        cursor = await conn.execute("SELECT 1 FROM admins WHERE user_id = ?", (user_id,))
        result = await cursor.fetchone()
        return result is not None

# Обновление кеша вопросов-ответов
async def update_qa_cache():
    global qa_cache, vectorizer_fitted, qa_vectors
    
    async with db_lock:
        conn = await get_db()
        cursor = await conn.execute("SELECT question, answer FROM qa WHERE answer IS NOT NULL")
        qa_cache = await cursor.fetchall()
        
        logger.info(f"Обновление ИИ-кеша: загружено {len(qa_cache)} вопрос-ответов")
        
        if qa_cache:
            questions = [q for q, a in qa_cache]
            try:
                qa_vectors = vectorizer.fit_transform(questions)
                vectorizer_fitted = True
                logger.info("Векторизатор TF-IDF успешно обучен")
            except ValueError as e:
                vectorizer_fitted = False
                logger.error(f"Ошибка обучения векторизатора: {e}")
        else:
            vectorizer_fitted = False
            logger.warning("Нет данных для обучения векторизатора")

# Функция для ИИ-ответов
async def ai_generate_answer(question: str) -> str:
    global vectorizer_fitted, qa_vectors
    
    if not vectorizer_fitted or not qa_cache:
        logger.info("ИИ-кеш не готов, пропуск генерации ответа")
        return None
        
    try:
        input_vector = vectorizer.transform([question])
        similarity_matrix = cosine_similarity(input_vector, qa_vectors)
        best_match_idx = np.argmax(similarity_matrix)
        best_score = similarity_matrix[0, best_match_idx]
        
        logger.debug(f"Схожесть вопроса '{question[:30]}...': {best_score:.2f}")
        
        if best_score >= AI_THRESHOLD:
            best_match = qa_cache[best_match_idx]
            async with db_lock:
                conn = await get_db()
                await conn.execute('''
                    UPDATE qa 
                    SET use_count = use_count + 1, 
                        last_used = CURRENT_TIMESTAMP 
                    WHERE question = ?
                ''', (best_match[0],))
                await conn.commit()
            return best_match[1]
        else:
            logger.info(f"ИИ не нашел подходящего ответа: схожесть {best_score:.2f} < {AI_THRESHOLD}")
    except Exception as e:
        logger.error(f"Ошибка ИИ-генерации: {e}")
    return None

# Поиск ответа в базе
async def find_answer(question: str) -> str:
    async with db_lock:
        conn = await get_db()
        cursor = await conn.execute("SELECT question, answer FROM qa WHERE answer IS NOT NULL")
        results = await cursor.fetchall()
        
        if not results:
            return None
        
        best_match = None
        best_ratio = 0
        
        clean_question = re.sub(r'[^\w\s]', '', question.lower()).strip()
        
        for q, a in results:
            clean_q = re.sub(r'[^\w\s]', '', q.lower()).strip()
            ratio = SequenceMatcher(None, clean_question, clean_q).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = (q, a)
        
        if best_ratio >= 0.7 and best_match:
            if not best_match[1] or best_match[1].strip() == "":
                logger.error("Найден пустой ответ в базе данных!")
                return None
            
            try:
                await conn.execute('''
                    UPDATE qa 
                    SET use_count = use_count + 1, 
                        last_used = CURRENT_TIMESTAMP 
                    WHERE question = ?
                ''', (best_match[0],))
                await conn.commit()
            except Exception as e:
                logger.error(f"Ошибка обновления счетчика: {e}")
            
            return best_match[1]
        return None

# Добавление вопроса в базу
async def add_question(question: str, user_id: int) -> int:
    try:
        async with db_lock:
            conn = await get_db()
            try:
                cursor = await conn.execute(
                    "INSERT INTO qa (question, user_id) VALUES (?, ?)", 
                    (question, user_id)
                )
                await conn.commit()
                return cursor.lastrowid
            except aiosqlite.IntegrityError:
                cursor = await conn.execute("SELECT id FROM qa WHERE question = ?", (question,))
                result = await cursor.fetchone()
                if result:
                    return result[0]
                return None
    except Exception as e:
        logger.error(f"Ошибка добавления вопроса: {e}")
        return None

# Добавление ответа в базу
async def add_answer(q_id: int, answer: str, answered_by: str = "admin"):
    try:
        async with db_lock:
            conn = await get_db()
            await conn.execute(
                "UPDATE qa SET answer = ?, answered_by = ? WHERE id = ?", 
                (answer, answered_by, q_id)
            )
            await conn.commit()
        await update_qa_cache()
    except Exception as e:
        logger.error(f"Ошибка добавления ответа: {e}")

# Логирование обратной связи
async def log_feedback(user_id: int, question: str, answer: str, feedback_type: str):
    try:
        async with db_lock:
            conn = await get_db()
            await conn.execute(
                "INSERT INTO feedback (user_id, question, answer, feedback_type) VALUES (?, ?, ?, ?)",
                (user_id, question, answer, feedback_type)
            )
            await conn.commit()
    except Exception as e:
        logger.error(f"Error logging feedback: {e}")

# Отправка уведомления админам
async def notify_admins(context: ContextTypes.DEFAULT_TYPE, question: str, user_id: int, q_id: int):
    try:
        async with db_lock:
            conn = await get_db()
            cursor = await conn.execute("SELECT user_id FROM admins")
            admins = [row[0] for row in await cursor.fetchall()]
    except Exception as e:
        logger.error(f"Ошибка получения списка админов: {e}")
        return
    
    message = (
        "🔔 *Новый вопрос!*\n\n"
        f"🆔 ID: `{q_id}`\n"
        f"❓ Вопрос: _{question}_\n"
        f"👤 Пользователь: `{user_id}`\n\n"
        "Ответьте командой:\n"
        "`/answer {ID} [ваш ответ]`"
    )
    
    for admin_id in admins:
        try:
            await context.bot.send_message(
                chat_id=admin_id,
                text=message,
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Ошибка уведомления админа {admin_id}: {e}")

# Обработчики команд
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [[KeyboardButton("🚀 Начать работу")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "Добро пожаловать! Нажмите кнопку ниже, чтобы начать работу с ботом.",
        reply_markup=reply_markup
    )
    context.user_data.clear()

async def start_work(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("❓ Задать вопрос")],
        [KeyboardButton("🔚 Завершить сеанс")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "Вы в главном меню. Выберите действие:",
        reply_markup=reply_markup
    )
    # Сбрасываем состояние ожидания вопроса
    if 'awaiting_question' in context.user_data:
        del context.user_data['awaiting_question']

async def handle_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    if text == "🚀 Начать работу":
        await start_work(update, context)
    elif text == "❓ Задать вопрос":
        # Показываем только кнопку завершения сеанса
        reply_markup = ReplyKeyboardMarkup([[KeyboardButton("🔚 Завершить сеанс")]], resize_keyboard=True)
        await update.message.reply_text(
            "Пожалуйста, введите ваш вопрос:\n\n"
            "Администратор проверит его и ответит в ближайшее время.",
            reply_markup=reply_markup
        )
        context.user_data['awaiting_question'] = True
    elif text == "🔚 Завершить сеанс":
        await end_session(update, context)

async def end_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Сбрасываем состояние ожидания вопроса
    if 'awaiting_question' in context.user_data:
        del context.user_data['awaiting_question']
    
    keyboard = [[KeyboardButton("🚀 Начать работу")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "✅ Сеанс завершен. Нажмите кнопку ниже, чтобы начать заново.",
        reply_markup=reply_markup
    )

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get('awaiting_question'):
        question = update.message.text
        user_id = update.message.from_user.id
        
        # Поиск точного ответа
        answer = await find_answer(question)
        answer_source = "Ответ из базы знаний"
        
        # ИИ-генерация если не найдено
        if not answer:
            answer = await ai_generate_answer(question)
            answer_source = "ИИ-ответ на основе похожих вопросов"
        
        if answer:
            response = f"🤖 {answer_source}:\n\n{answer}\n\nЕсли ответ не подходит, задайте вопрос более подробно."
            await update.message.reply_text(response)

            if answer_source.startswith("ИИ"):
                context.user_data['last_ai_question'] = question
                context.user_data['last_ai_answer'] = answer
                keyboard = [
                    [KeyboardButton("👍 Ответ подходит")],
                    [KeyboardButton("👎 Ответ не подходит")]
                ]
                reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
                await update.message.reply_text(
                    "Пожалуйста, оцените, насколько ответ был полезен:",
                    reply_markup=reply_markup
                )
            else:
                # После ответа из базы показываем основное меню
                await start_work(update, context)
        else:
            q_id = await add_question(question, user_id)
            if q_id:
                await notify_admins(context, question, user_id, q_id)
                await update.message.reply_text(
                    "✅ Ваш вопрос принят! Администратор проверит его и ответит в ближайшее время."
                )
                # Возвращаем основное меню
                await start_work(update, context)
        # Сбрасываем флаг ожидания вопроса
        if 'awaiting_question' in context.user_data:
            del context.user_data['awaiting_question']
        return

async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    feedback = update.message.text
    user_id = update.message.from_user.id
    
    if 'last_ai_question' in context.user_data and 'last_ai_answer' in context.user_data:
        question = context.user_data['last_ai_question']
        answer = context.user_data['last_ai_answer']
        feedback_type = "positive" if feedback == "👍 Ответ подходит" else "negative"
        await log_feedback(user_id, question, answer, feedback_type)
        
        del context.user_data['last_ai_question']
        del context.user_data['last_ai_answer']
        
        await update.message.reply_text(
            "Спасибо за вашу оценку! Это поможет улучшить бота.",
            reply_markup=ReplyKeyboardRemove()
        )
        
        # После оценки показываем основное меню
        await start_work(update, context)
        
        if feedback_type == "negative":
            await update.message.reply_text(
                "Извините, что ответ не подошел. Я передал ваш вопрос администратору."
            )
            q_id = await add_question(question, user_id)
            if q_id:
                await notify_admins(context, question, user_id, q_id)
    else:
        await update.message.reply_text(
            "Контекст ответа утерян, но спасибо за вашу оценку!",
            reply_markup=ReplyKeyboardRemove()
        )
        # Возвращаем в главное меню
        await start_work(update, context)

# Режим администратора (скрыт от обычных пользователей)
async def admin_login(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    if await is_admin(user_id):
        await admin_menu(update, context)
        return ADMIN_MODE
    await update.message.reply_text(
        "🔒 Введите пароль для входа в режим администратора:",
        reply_markup=ReplyKeyboardRemove()
    )
    return ADMIN_LOGIN

async def admin_auth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    password = update.message.text
    user_id = update.message.from_user.id
    
    if password == ADMIN_PASSWORD:
        username = update.message.from_user.username or "без username"
        try:
            async with db_lock:
                conn = await get_db()
                await conn.execute(
                    "INSERT OR IGNORE INTO admins (user_id, username) VALUES (?, ?)", 
                    (user_id, username)
                )
                await conn.commit()
        except Exception as e:
            logger.error(f"Ошибка добавления администратора: {e}")
        
        await update.message.reply_text("✅ Успешный вход в режим администратора!")
        await admin_menu(update, context)
        return ADMIN_MODE
    else:
        await update.message.reply_text("❌ Неверный пароль. Попробуйте снова.")
        return ADMIN_LOGIN

async def admin_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    if not await is_admin(user_id):
        await update.message.reply_text("❌ У вас нет прав администратора.")
        return ConversationHandler.END
    
    keyboard = [
        [KeyboardButton("📊 Статистика")],
        [KeyboardButton("❓ Неотвеченные вопросы")],
        [KeyboardButton("🧹 Очистить неотвеченные")],
        [KeyboardButton("🔄 Обновить ИИ-кеш")],
        [KeyboardButton("🔙 Завершить сеанс администратора")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "👑 Вы в режиме администратора. Выберите действие:",
        reply_markup=reply_markup
    )
    return ADMIN_MODE

async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        async with db_lock:
            conn = await get_db()
            
            # Всего вопросов
            cursor = await conn.execute("SELECT COUNT(*) FROM qa")
            total_questions = (await cursor.fetchone())[0]
            
            # Отвеченных вопросов
            cursor = await conn.execute("SELECT COUNT(*) FROM qa WHERE answer IS NOT NULL")
            answered_questions = (await cursor.fetchone())[0]
            
            # Всего отзывов
            cursor = await conn.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = (await cursor.fetchone())[0]
            
            # Положительных отзывов
            cursor = await conn.execute("SELECT COUNT(*) FROM feedback WHERE feedback_type = 'positive'")
            positive_feedback = (await cursor.fetchone())[0]
            
            # Отрицательных отзывов
            cursor = await conn.execute("SELECT COUNT(*) FROM feedback WHERE feedback_type = 'negative'")
            negative_feedback = (await cursor.fetchone())[0]
            
            # Популярные вопросы
            cursor = await conn.execute("""
                SELECT question, use_count 
                FROM qa 
                WHERE use_count > 0 
                ORDER BY use_count DESC 
                LIMIT 5
            """)
            popular_questions = await cursor.fetchall()
            
            # Рассчет процента отвеченных вопросов
            answered_percent = 0.0
            if total_questions > 0:
                answered_percent = (answered_questions / total_questions) * 100
            
            report = (
                f"📊 *Статистика бота*\n\n"
                f"• Всего вопросов: `{total_questions}`\n"
                f"• Отвечено: `{answered_questions}` ({answered_percent:.1f}%)\n"
                f"• Положительных оценок: `{positive_feedback}`\n"
                f"• Отрицательных оценок: `{negative_feedback}`\n\n"
            )
            
            if popular_questions:
                report += "🔝 *Топ-5 популярных вопросов:*\n"
                for i, (question, count) in enumerate(popular_questions, 1):
                    # Безопасное обрезание длинных вопросов
                    truncated_question = (question[:50] + '...') if len(question) > 50 else question
                    report += f"{i}. {truncated_question} - {count} раз\n"
            else:
                report += "🔝 Популярных вопросов пока нет\n"
            
            await update.message.reply_text(report, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}", exc_info=True)
        await update.message.reply_text("❌ Ошибка при получении статистики. Подробности в логах.")

async def unanswered_questions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        async with db_lock:
            conn = await get_db()
            cursor = await conn.execute("""
                SELECT id, question, user_id, created_at 
                FROM qa 
                WHERE answer IS NULL 
                ORDER BY created_at DESC
            """)
            questions = await cursor.fetchall()
            
            if not questions:
                await update.message.reply_text("🎉 Все вопросы отвечены!")
                return
            
            report = "❓ *Неотвеченные вопросы:*\n\n"
            for q_id, question, asker_id, created_at in questions:
                # Форматирование даты
                created_date = created_at.split('.')[0] if '.' in created_at else created_at
                report += (
                    f"🆔 ID: `{q_id}`\n"
                    f"👤 Пользователь: `{asker_id}`\n"
                    f"🕒 Дата: {created_date}\n"
                    f"❓ Вопрос: _{question}_\n\n"
                )
            
            report += "\nДля ответа используйте команду:\n`/answer ID_вопроса ваш ответ`"
            await update.message.reply_text(report, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Ошибка получения вопросов: {e}")
        await update.message.reply_text("❌ Ошибка при получении вопросов")

async def refresh_cache(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update_qa_cache()
    await update.message.reply_text("🔄 Кеш ИИ-модели успешно обновлен!")

async def answer_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "❌ Неверный формат команды. Используйте:\n"
            "`/answer ID_вопроса ваш ответ`"
        )
        return
    
    try:
        q_id = int(context.args[0])
        answer = " ".join(context.args[1:])
        
        async with db_lock:
            conn = await get_db()
            cursor = await conn.execute("SELECT question, user_id FROM qa WHERE id = ?", (q_id,))
            question_data = await cursor.fetchone()
            
            if not question_data:
                await update.message.reply_text(f"❌ Вопрос с ID {q_id} не найден.")
                return
                
            question_text, asker_id = question_data
        
        await add_answer(q_id, answer)
        
        if asker_id:
            try:
                await context.bot.send_message(
                    chat_id=asker_id,
                    text=(
                        f"📬 Получен ответ на ваш вопрос (ID: {q_id})!\n\n"
                        f"❓ Ваш вопрос: {question_text}\n"
                        f"💬 Ответ администратора: {answer}"
                    )
                )
                await update.message.reply_text(f"✅ Ответ отправлен пользователю {asker_id}")
            except telegram.error.Unauthorized:
                await update.message.reply_text("❌ Пользователь заблокировал бота.")
            except Exception as e:
                await update.message.reply_text(f"❌ Ошибка отправки: {e}")
        else:
            await update.message.reply_text(f"✅ Ответ сохранен для вопроса ID {q_id}")
        
    except ValueError:
        await update.message.reply_text("❌ Неверный формат ID вопроса.")
    except Exception as e:
        logger.error(f"Ошибка при сохранении ответа: {e}")
        await update.message.reply_text("❌ Ошибка при сохранении ответа")

async def clean_unanswered_questions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    if not await is_admin(user_id):
        await update.message.reply_text("❌ У вас нет прав администратора.")
        return
    
    # Запрашиваем подтверждение
    keyboard = [["✅ Да, очистить"], ["❌ Нет, отменить"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    await update.message.reply_text(
        "⚠️ Вы уверены, что хотите удалить ВСЕ неотвеченные вопросы? Это действие нельзя отменить.\n\n"
        "Пожалуйста, подтвердите:",
        reply_markup=reply_markup
    )
    context.user_data['awaiting_clean_confirm'] = True

async def handle_clean_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    user_id = update.message.from_user.id
    
    if not await is_admin(user_id):
        await update.message.reply_text("❌ У вас нет прав администратора.")
        return
    
    if 'awaiting_clean_confirm' in context.user_data:
        if text == "✅ Да, очистить":
            try:
                async with db_lock:
                    conn = await get_db()
                    # Получаем количество перед удалением
                    cursor = await conn.execute("SELECT COUNT(*) FROM qa WHERE answer IS NULL")
                    count = (await cursor.fetchone())[0]
                    
                    if count == 0:
                        await update.message.reply_text("ℹ️ Неотвеченных вопросов для очистки не найдено.")
                        return
                    
                    # Выполняем удаление
                    await conn.execute("DELETE FROM qa WHERE answer IS NULL")
                    await conn.commit()
                
                await update.message.reply_text(f"🧹 Успешно удалено {count} неотвеченных вопросов!")
            except Exception as e:
                logger.error(f"Ошибка очистки вопросов: {e}")
                await update.message.reply_text("❌ Произошла ошибка при очистке вопросов.")
        else:
            await update.message.reply_text("❌ Очистка отменена.")
        
        # Удаляем флаг ожидания подтверждения
        del context.user_data['awaiting_clean_confirm']
        await admin_menu(update, context)

async def cancel_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("🔒 Режим администратора завершен.", reply_markup=ReplyKeyboardRemove())
    await start(update, context)
    return ConversationHandler.END

# Настройка команд бота (скрываем команду admin из меню)
async def post_init(application: Application):
    await init_db()
    await application.bot.set_my_commands([
        BotCommand("start", "Запустить бота"),
        BotCommand("help", "Помощь")
    ])
    await update_qa_cache()
    logger.info("ИИ-модель инициализирована")

async def on_shutdown(application: Application):
    await close_db()
    logger.info("Бот остановлен. Соединения с БД закрыты.")

# Основная функция
def main() -> None:
    application = Application.builder()\
        .token(TOKEN)\
        .post_init(post_init)\
        .post_stop(on_shutdown)\
        .build()
    
    admin_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('admin', admin_login)],
        states={
            ADMIN_LOGIN: [MessageHandler(filters.TEXT & ~filters.COMMAND, admin_auth)],
            ADMIN_MODE: [
                MessageHandler(filters.Regex(r'^📊 Статистика$'), admin_stats),
                MessageHandler(filters.Regex(r'^❓ Неотвеченные вопросы$'), unanswered_questions),
                MessageHandler(filters.Regex(r'^🧹 Очистить неотвеченные$'), clean_unanswered_questions),
                MessageHandler(filters.Regex(r'^🔄 Обновить ИИ-кеш$'), refresh_cache),
                MessageHandler(filters.Regex(r'^🔙 Завершить сеанс администратора$'), cancel_admin),
                CommandHandler('answer', answer_question),
                # Обработчик подтверждения очистки
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_clean_confirm)
            ]
        },
        fallbacks=[CommandHandler('cancel', cancel_admin)],
        allow_reentry=True
    )
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(admin_conv_handler)
    
    application.add_handler(MessageHandler(filters.Regex(r'^🚀 Начать работу$'), handle_menu))
    application.add_handler(MessageHandler(filters.Regex(r'^❓ Задать вопрос$'), handle_menu))
    application.add_handler(MessageHandler(filters.Regex(r'^🔚 Завершить сеанс$'), handle_menu))
    application.add_handler(MessageHandler(filters.Regex(r'^👍 Ответ подходит$|^👎 Ответ не подходит$'), handle_feedback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    
    logger.info("Бот запущен")
    
    try:
        application.run_polling()
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    finally:
        # Закрытие соединения при выходе
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.run_until_complete(close_db())

if __name__ == "__main__":
    main()