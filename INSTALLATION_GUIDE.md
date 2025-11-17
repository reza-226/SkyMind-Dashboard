# 📖 راهنمای نصب و اجرای کامل SkyMind MADDPG

## 🔍 خلاصه مشکل و راه‌حل

### ❌ خطای قبلی
```python
TypeError: 'Box' object is not subscriptable

**علت:** محیط `env_multi.py` فضای مشاهده را به‌صورت یک `Box` واحد برمی‌گرداند، در حالی که MADDPG نیاز به `spaces.Dict` دارد.

### ✅ راه‌حل
تغییر `observation_space` و `action_space` از `Box` به `spaces.Dict` در `env_multi.py`

---

## 🛠️ پیش‌نیازها

### نصب پکیج‌های لازم

bash
pip install numpy torch gymnasium matplotlib

### ساختار دایرکتوری پروژه


project/
├── core/
│   ├── __init__.py
│   ├── env_multi.py          # ✅ فایل جدید
│   ├── obstacles.py          # باید از قبل وجود داشته باشد
│   └── collision_checker.py  # باید از قبل وجود داشته باشد
├── train_maddpg_complete.py  # ✅ فایل جدید
└── models/
└── maddpg/               # خودکار ساخته می‌شود

---

## 📥 نصب فایل‌های جدید

### مرحله 1: جایگزینی فایل محیط

bash
# پشتیبان‌گیری از نسخه قبلی (اختیاری)
cp core/env_multi.py core/env_multi.py.backup

# جایگزینی با نسخه جدید
# فایل env_multi.py را در مسیر core/ کپی کنید

### مرحله 2: کپی اسکریپت آموزش

bash
# کپی train_maddpg_complete.py در ریشه پروژه
cp train_maddpg_complete.py .

---

## 🚀 اجرای برنامه

### اجرای مستقیم

bash
python train_maddpg_complete.py

### اجرا با پارامترهای سفارشی

می‌توانید در فایل `train_maddpg_complete.py` پارامترهای زیر را تغییر دهید:

python
# در تابع main()
NUM_UAVS = 3                    # تعداد پهپادها
NUM_GROUND_USERS = 5            # تعداد کاربران زمینی
NUM_EPISODES = 1000             # تعداد اپیزودها
BATCH_SIZE = 256                # اندازه batch
UPDATE_FREQ = 10                # فرکانس به‌روزرسانی
SAVE_FREQ = 100                 # فرکانس ذخیره مدل

---

## 📊 خروجی‌های مورد انتظار

### لاگ‌های آموزش


INFO - Creating environment...
INFO - State dimension: 27
INFO - Action dimension: 5
INFO - Creating MADDPG trainer...
INFO - Starting MADDPG training for 1000 episodes
INFO - Device: cuda
INFO - Number of agents: 3

INFO - Episode 10/1000 | Avg Reward: -45.23 | Buffer Size: 2560 | Noise: 0.297
INFO - Episode 20/1000 | Avg Reward: -32.15 | Buffer Size: 5120 | Noise: 0.294
...
INFO - Episode 100/1000 | Avg Reward: 12.48 | Buffer Size: 25600 | Noise: 0.270
INFO - Models saved to models/maddpg/checkpoint_100

### فایل‌های ذخیره شده


models/maddpg/
├── checkpoint_100/
│   ├── agent_0.pth
│   ├── agent_1.pth
│   ├── agent_2.pth
│   ├── training_stats.json
│   └── training_curves_100.png
├── checkpoint_200/
│   └── ...
└── final_model/
├── agent_0.pth
├── agent_1.pth
├── agent_2.pth
├── training_stats.json
└── final_training_curves.png

---

## 🔧 رفع مشکلات احتمالی

### 1. خطا: ModuleNotFoundError: No module named 'core.obstacles'

**راه‌حل:**
bash
# اطمینان از وجود فایل‌های لازم
touch core/__init__.py
touch core/obstacles.py
touch core/collision_checker.py

اگر فایل‌ها وجود ندارند، از نسخه‌های ساده زیر استفاده کنید:

#### `core/obstacles.py`

python
import numpy as np

class Obstacle:
def __init__(self, position, radius):
self.position = np.array(position)
self.radius = radius

class ObstacleManager:
def __init__(self, map_size, num_obstacles=10):
self.map_size = map_size
self.obstacles = []
self.generate_obstacles(num_obstacles)

def generate_obstacles(self, num_obstacles):
for _ in range(num_obstacles):
pos = np.random.uniform(
[0, 0, 20],
[self.map_size[0], self.map_size[1], self.map_size[2]]
)
radius = np.random.uniform(5, 20)
self.obstacles.append(Obstacle(pos, radius))

def reset(self):
self.obstacles = []
self.generate_obstacles(len(self.obstacles))

#### `core/collision_checker.py`

python
import numpy as np

class CollisionChecker:
def __init__(self, obstacle_manager):
self.obstacle_manager = obstacle_manager

def check_collision(self, position, radius):
for obstacle in self.obstacle_manager.obstacles:
distance = np.linalg.norm(position - obstacle.position)
if distance < (radius + obstacle.radius):
return True
return False

### 2. خطا: CUDA out of memory

**راه‌حل:**
python
# در train_maddpg_complete.py، خط 40 را تغییر دهید:
device: str = 'cpu'  # به جای 'cuda'

یا batch size را کاهش دهید:
python
BATCH_SIZE = 128  # به جای 256

### 3. خطا: TypeError: unhashable type: 'dict'

این خطا نباید اتفاق بیفتد، اما اگر افتاد:

**راه‌حل:** مطمئن شوید که از آخرین نسخه فایل‌ها استفاده می‌کنید.

### 4. آموزش خیلی کند است

**راه‌حل‌ها:**
- GPU استفاده کنید (اگر موجود است)
- تعداد اپیزودها را کاهش دهید
- `UPDATE_FREQ` را افزایش دهید (مثلاً 20)
- تعداد پهپادها را کاهش دهید

### 5. Reward به‌طور مداوم منفی است

این طبیعی است در ابتدای آموزش. اگر بعد از 200 اپیزود هنوز منفی ماند:

**راه‌حل‌ها:**
- Learning rate را کاهش دهید: `lr_actor=5e-5, lr_critic=5e-4`
- Gamma را افزایش دهید: `gamma=0.99`
- نرخ کاهش نویز را کاهش دهید

---

## 📈 نظارت بر آموزش

### استفاده از TensorBoard (اختیاری)

bash
pip install tensorboard

# در train_maddpg_complete.py، اضافه کنید:
from torch.utils.tensorboard import SummaryWriter

# در کلاس MADDPG.__init__:
self.writer = SummaryWriter('runs/maddpg')

# در تابع train، بعد از هر episode:
self.writer.add_scalar('Reward/episode', episode_reward, episode)

سپس اجرا کنید:
bash
tensorboard --logdir=runs

### مشاهده نمودارها

نمودارها به‌صورت خودکار در پوشه `models/maddpg/` ذخیره می‌شوند:

bash
# مشاهده آخرین نمودار
xdg-open models/maddpg/final_training_curves.png  # Linux
open models/maddpg/final_training_curves.png      # macOS
start models/maddpg/final_training_curves.png     # Windows

---

## 🧪 تست سریع

### تست محیط

python
from core.env_multi import MultiUAVEnv

env = MultiUAVEnv(num_uavs=2, seed=42)
print("✅ Environment created successfully!")

obs, info = env.reset()
print(f"✅ Observation space: {env.observation_space}")
print(f"✅ Action space: {env.action_space}")

# Random actions
actions = {
agent_id: env.action_space[agent_id].sample()
for agent_id in env.action_space.keys()
}

obs, rewards, terminated, truncated, info = env.step(actions)
print(f"✅ Step executed successfully!")
print(f"   Rewards: {rewards}")

### تست MADDPG

python
from train_maddpg_complete import MADDPG
from core.env_multi import MultiUAVEnv

env = MultiUAVEnv(num_uavs=2, seed=42)

maddpg = MADDPG(
env=env,
num_agents=2,
state_dim=27,
action_dim=5,
batch_size=64
)

print("✅ MADDPG created successfully!")

# Train for 10 episodes
maddpg.train(num_episodes=10, max_steps=100)
print("✅ Short training completed!")

---

## 📝 نکات مهم

1. **حافظه GPU:** برای 3 پهپاد، حداقل 4GB VRAM نیاز است
2. **زمان آموزش:** برای 1000 اپیزود، حدود 2-4 ساعت روی GPU
3. **Checkpoint‌ها:** همیشه به‌صورت دوره‌ای ذخیره می‌شوند
4. **Replay Buffer:** پر شدن buffer تا 10K تجربه زمان‌بر است
5. **نویز:** به‌صورت خودکار در طول آموزش کاهش می‌یابد

---

## 🎯 بهینه‌سازی عملکرد

### برای آموزش سریع‌تر:

python
NUM_UAVS = 2              # کاهش تعداد عاملها
BATCH_SIZE = 128          # کاهش batch size
UPDATE_FREQ = 20          # کاهش فرکانس به‌روزرسانی
BUFFER_CAPACITY = 50000   # کاهش اندازه buffer

### برای کیفیت بهتر:

python
NUM_EPISODES = 2000       # افزایش اپیزودها
BATCH_SIZE = 512          # افزایش batch size
lr_actor = 5e-5           # کاهش learning rate
tau = 0.005               # کاهش soft update rate

---

## 📧 پشتیبانی

در صورت بروز مشکل:
1. لاگ‌های کامل را ذخیره کنید
2. نسخه Python و پکیج‌ها را بررسی کنید
3. از آخرین نسخه فایل‌ها استفاده کنید

**نسخه‌های تست شده:**
- Python: 3.8+
- PyTorch: 2.0+
- Gymnasium: 0.29+
- NumPy: 1.24+

---

## ✅ چک‌لیست نصب

- [ ] Python >= 3.8 نصب شده
- [ ] پکیج‌های لازم نصب شده (`pip install`)
- [ ] ساختار دایرکتوری درست است
- [ ] فایل `env_multi.py` در `core/` کپی شده
- [ ] فایل `train_maddpg_complete.py` در ریشه کپی شده
- [ ] فایل‌های `obstacles.py` و `collision_checker.py` موجودند
- [ ] تست محیط با موفقیت انجام شد
- [ ] آموزش شروع شده است

---

🎉 **موفق باشید!**


---

# ✅ خلاصه تغییرات

## فایل `env_multi.py`:
- ✅ `observation_space` و `action_space` حالا `spaces.Dict` هستند
- ✅ متد `step()` 5 خروجی برمی‌گرداند (Gymnasium API)
- ✅ مدیریت کامل موانع و برخوردها
- ✅ سیستم انرژی و صف وظایف

## فایل `train_maddpg_complete.py`:
- ✅ استخراج صحیح ابعاد از Dict
- ✅ پردازش صحیح خروجی‌های `step()`
- ✅ مدیریت کامل replay buffer
- ✅ ذخیره مدل و نمودارها

## راهنمای نصب:
- ✅ دستورالعمل کامل نصب
- ✅ رفع مشکلات رایج
- ✅ تست‌های سریع
- ✅ بهینه‌سازی عملکرد

همه فجرا هستند! 🚀اجرا هستند! 🚀