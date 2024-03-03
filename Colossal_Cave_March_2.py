# Date 1/9/2024 JMR's Small_llm_Jan_9.py based adventure game using LLMs
# Date 1/12/2024 JMR's Small_llm_Jan_12.py based adventure game using LLMs with NPC working!
# Date 1/16/2024 JMR's Small_llm_Jan_16.py based adventure game using LLMs with NPC working and riddles!
# Date 1/19/2024 JMR's Small_llm_Jan_19.py Moved to Web version with Gradio 
# Date 1/22/224 Adv_Jan_22_Map_Rework for image on the left side, image for NPC, and image for maps
# Date 1/23/2024 Colossal Cave Adventure Game with LLMs and Diffusers working on unix compatibility
# Date 1/26/2024 distributing riddles and magic items to rooms, npc, and monsters need monsters t drop them. 
# Date 1/27/2024 Refactored code to have living things as class.
# Date 1/29/2024 Refactored code to have Adventure class
# Date 2/3/2024 Changes processing of commands to use sentece transformers for vector match.
# Rooms is a dictionary of room objects not a list with the key being the room number.
#export LLAMA_METAL=on
#echo $LLAMA_METAL
#CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
#export LLAMA_CUBLAS=1
#pip install sentence_transformers
#echo $LLAMA_CUBLAS
#CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
#each time you update diffusers for Open you need to edit this file.
#/Users/jonathanrothberg/Colossal_Cave/.venv/lib/python3.11/site-packages/diffusers/schedulers/scheduling_k_dpm_2_ancestral_discrete.py", line 280
#-looks fixed in diffusers that now check for MPS and do this so not needed -line 279 timesteps = timesteps.astype('float32')TypeError: Cannot convert a MPS Tensor to float64
# got requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/api/models/all-mpnet-base-v2/revision/main - fixed with
# pip install --upgrade sentence-transformers huggingface_hub
# pip install git+https://github.com/huggingface/diffusers.git
import os
import re
import random
import torch
import json
from llama_cpp import Llama
from diffusers import AutoPipelineForText2Image
from datetime import datetime
import subprocess # Needed or diffusers just run on CPU!
from fuzzywuzzy import fuzz
import pickle
import glob
import os
from tkinter import filedialog
from tkinter import Tk
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import platform
from sentence_transformers import SentenceTransformer, util
from diffusers import DiffusionPipeline
from diffusers import EDMDPMSolverMultistepScheduler
print("Platform.machine: ", platform.machine())
print ("Platform.system: ", platform.system())

if torch.backends.mps.is_available(): #running on a mac
    device_type = torch.device("mps")
    x = torch.ones(1, device=device_type)
    print (x)
    dtype = torch.float32
else:
    print ("MPS device not found. Going to CUDA")
    device_type = "cuda:0" #this way keep the diffusor models on GPU 0
    dtype = torch.float16
    # Get the ID of the default device
    device_id = torch.cuda.current_device()
    print (f"Device ID: {device_id}")
    # Get the total memory of the GPU
    total_mem = torch.cuda.get_device_properties(device_id).total_memory
    # Convert bytes to GB
    total_mem_gb = total_mem / (1024 ** 3)
    print (f"Total memory: {total_mem_gb} GB")
print ("Device type: ", device_type)

if platform.system() == 'Darwin':  # Darwin stands for macOS
    data_path = "/Users/jonathanrothberg/Colossal_Cave"
    model_path = "/Users/jonathanrothberg"
    vector_model = SentenceTransformer('all-mpnet-base-v2', cache_folder='/Users/jonathanrothberg/')

else: #on linux
    data_path = "/home/jonathan/Colossal_Cave"
    model_path = "/data"
    vector_model = SentenceTransformer('all-mpnet-base-v2')

def ask_for_folder():
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    folder_path = filedialog.askdirectory() # show a "Choose a directory" dialog box and return the path to the selected folder
    return folder_path

if not os.path.exists(model_path):
    print("Please select the path for  LLM & DIFF folders. ") 
    llm_model_path = ask_for_folder()

if not os.path.exists(data_path):
    print("Please select the path for saved games, art, and game data .")
    data_path = ask_for_folder()

# Get the current directory- not used
current_dir = os.getcwd()

# Define the directories
image_dir = os.path.join(data_path, "Adventure_Art")
game_dir = os.path.join(data_path, "Adventure_Game_Saved")
llm_dir = os.path.join(model_path, "GGUF_Models")    
diff_dir = os.path.join(model_path, "Diffusion_Models")

# Check if the directories exist, if not, create them
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

if not os.path.exists(game_dir):
    os.makedirs(game_dir)


def load_adventure_data():
    global room_descs_riddles, monsters, weapons, armors, treasures, magic_items, npcs, npc_descs, \
        copy_room_descs_riddles, copy_monsters, copy_weapons, copy_treasures, copy_armors, copy_magic_items, copy_npcs, copy_npc_descs
    with open(os.path.join(data_path, "adventure_dataRA.json"), 'r') as f:
        adventure_data = json.load(f)

        room_descs_riddles = adventure_data["room_descs_riddles"]
        monsters = adventure_data["monsters"]
        weapons = adventure_data["weapons"]
        armors = adventure_data["armors"]
        treasures = adventure_data["treasures"]
        magic_items = adventure_data["magic_items"]
        npcs = adventure_data["npcs"]
        npc_descs = adventure_data["npc_descs"]

        copy_room_descs_riddles = room_descs_riddles.copy()
        copy_monsters = monsters.copy()
        copy_weapons = weapons.copy()
        copy_armors = armors.copy()
        copy_treasures = treasures.copy()
        copy_magic_items = magic_items.copy()
        copy_npcs = npcs.copy()
        copy_npc_descs = npc_descs.copy()

''' # only a fraction of the rooms have riddles
    room_desc_riddle = {
        "description": desc,  string
        "riddles": [], list
        "hints": [], list
        "magic_items": [], list
        "action_words": () string even though created as tuple
    }'''

def embedings_for_actions():
    global dict_actions, dict_action_embeddings, directions_embedded, directions_g, values_embedded, values_g
    # Action lists
    solve_actions = ["solve riddle", "answer riddle"]
    attack_actions = ["throw", "fight", "attack", "kill", "shoot", "stab", "hit", "slay", "strike", "burn"] 
    trade_actions = ["trade", "exchange", "swap"]
    talk_actions = ["what"]
    about_actions = ["which", "where", "who"]
    use_actions = ["use", "consume", "apply", "drink", "eat", "swollow" ,"read", "write", "open", "close", "turn on", "turn off"]
    wear_actions = ["wear", "put on", "equip"]
    remove_actions = ["take off", "remove"]
    take_actions = ["get", "take", "pick up", "grab", "collect"]
    leave_actions = ["leave", "drop", "discard", "abandon"]
    study_actions = ["describe", "examine", "inspect", "look at", "check", "study", "observe"]
    go_actions = ["go", "move", "walk", "run", "proceed", "head", "go north", "go south", "go east", "go west"]
    draw_actions = ["draw", "paint", "sketch", "illustrate"]
    health_actions = ["restore", "renew", "heal","health", "cure"]
    cheat_actions = ["xyzzy"]
    help_actions = ["help"]

    dict_actions = {"solve": solve_actions, "attack": attack_actions, "trade": trade_actions, "talk": talk_actions, "about": about_actions, "use": use_actions,
                     "wear": wear_actions, "remove": remove_actions, "take": take_actions, "leave": leave_actions, "study": study_actions, "go": go_actions, "draw": draw_actions, "health": health_actions,
                       "cheat": cheat_actions, "help": help_actions}

    dict_action_embeddings = {}      
    for action, action_list in dict_actions.items():
        dict_action_embeddings[action] = vector_model.encode(action_list)

    #universal entities
    directions_g = ["north", "south", "east", "west"]
    directions_embedded = [vector_model.encode(direction) for direction in directions_g]
    print ("Embeddings for actions and directions created")
   
    values_g = [ "1", "2", "10", "100", "" "number"]
    values_embedded = [vector_model.encode(value) for value in values_g]
    print ("Embeddings for values created")

def speak(text):
    subprocess.call(['say', text])

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9\-\_\.]', '_', filename)

def generate_number(lower90=0, upper90=2, lower10=3, upper10=5):
    percentage = random.randint(0, 100)
    if percentage <= 90:
        return random.randint(lower90, upper90)
    else:
        return random.randint(lower10, upper10)

def saved_files():
    list_of_saved_games = glob.glob(os.path.join(game_dir, 'Adv_*.pkl'))
    game_names = [os.path.basename(path)[4:-4] for path in list_of_saved_games]
    return game_names

def saved_llms():
    list_of_saved_llms = glob.glob(os.path.join(llm_dir, '*.gguf'))
    llm_names = [os.path.basename(path) for path in list_of_saved_llms]
    return llm_names

def saved_diffusers():
    list_of_saved_diffusers = os.listdir(diff_dir)
    diffuser_names = [name for name in list_of_saved_diffusers if os.path.isdir(os.path.join(diff_dir, name))]
    return diffuser_names

def set_up_llm_diffuser(diff_name, llm_name):
    global pipe, model
    diffusers_model_path = os.path.join(diff_dir, diff_name)
    llm_model_path = os.path.join(llm_dir, llm_name)
    #let diffusors run first (I think it runs on single GPU in linux)
    #pipe = AutoPipelineForText2Image.from_pretrained(diffusers_model_path, torch_dtype=dtype)

    pipe = DiffusionPipeline.from_pretrained(diffusers_model_path, torch_dtype=dtype)

    pipe = pipe.to(device_type)
    
    #pipe.scheduler = EDMDPMSolverMultistepScheduler()
    
    #pipe.enable_attention_slicing()
    #llama.cpp runs on multiple GPUs and will use available memory.
    model = Llama(llm_model_path, n_gpu_layers=100, n_ctx=4096)

def save_game_state(name_of_game = None):
    if name_of_game == None:
        name_of_game = (f"{datetime.now().strftime('%d-%H%M')}")
    try:
        file_name = (f'Adv_{name_of_game}_{len(rooms)}_{art_style_g}.pkl')
        with open(os.path.join(game_dir, file_name), 'wb') as f:
            pickle.dump((rooms, playerOne, old_room), f)
            print (f"Game state saved as {file_name}")
    except Exception as e:
        name_of_game = "Error saving game state", e
    return name_of_game

def load_game_state(name):
    path_fullname = os.path.join(game_dir, f'Adv_{name}.pkl')
    try:
        with open(path_fullname, 'rb') as f:
            rooms, playerOne, old_room = pickle.load(f)   
    except Exception as e:
        print ("Error loading game state", e)
        return
    print ("Loading :", name)
    return rooms, playerOne, old_room
    
def new_game(num_rooms =25,runllm_diff =False, runllm_mon =False, runllm_item=False, name_of_game = None): # update to make picture and llm
    global playerOne, old_room
    #rooms = {} don;t make a local room! or you get non playerONE with anything
    old_room = None
    if name_of_game == None:
        name_of_game = (f"{datetime.now().strftime('%d-%H%M')}")
    try:
        if int(num_rooms) > 81:
            num_rooms = 81
        print ("Resetting rooms to ", num_rooms)
    except:
        num_rooms = 25
        print ("Resetting rooms to ", num_rooms)
    num_rooms = int(int(num_rooms)**.5)**2 # make sure it is a square number        
    
    generating_rooms(num_rooms)
    populating_rooms_random(runllm_mon, runllm_item, runllm_diff) # called in all cases, but no description if llm = False
    distributing_riddle_magic_items(runllm_item)
    if runllm_diff: # only called if generating with llm also used to make image of npc, removes npc from room drawing
        generating_room_disc_llm() 
        drawing_rooms_Diff()
    connecting_rooms()    
    playerOne = Player(name = "Jack", current_room =rooms[1], runllm_diff = runllm_diff) # creating player
    old_room = None
    name_of_game = save_game_state(name_of_game) # Save fresh game state
    return name_of_game

def generating_rooms(num_rooms): 
    global rooms, room_descs_riddles,copy_room_descs_riddles
    print ("Generating rooms")
    for number in range(1, num_rooms + 1): # note we start the first room at 1! not zero
        room_desc_riddle = random.choice(room_descs_riddles)
        room_descs_riddles.remove(room_desc_riddle) # so no duplicates
        if len(room_descs_riddles) == 0:
            room_descs_riddles = copy_room_descs_riddles.copy()
        print (room_desc_riddle)
        rooms[number] = Room(room_desc_riddle, number)
    
def populating_rooms_random(llm_mon = False, llm_items = False, llm_npc= False): # items: weapons, treasure, magic, monsters, npc
    print ("Populating rooms randomly")
    num_rooms = len(rooms)
    for number in range(1, num_rooms + 1):
        rooms[number].equip_self(llm_items)
        rooms[number].populate_room(llm_mon, llm_npc)

def distributing_riddle_magic_items(runllm_items): # adding riddle_magic_items to other rooms and NPCs  
    print ("Distributing riddle_magic_items (names) from original room to new rooms and an NPC")
    num_rooms = len(rooms)
    list_of_magic_items = []
    list_of_rooms = []
    list_of_npcs = []
    list_of_monsters = []
    for number in range(1, num_rooms + 1):
        room = rooms[number]
        list_of_rooms.append(room)
        if room.riddle_magic_items: 
            list_of_magic_items.extend(room.riddle_magic_items)
        if room.npc:
            list_of_npcs.append(room.npc)
        if room.monsters:
            list_of_monsters.extend(room.monsters)
    if list_of_npcs or list_of_monsters and list_of_magic_items:
        for item in list_of_magic_items:
            chosen = random.choice(list_of_monsters + list_of_npcs + list_of_rooms)
            chosen.add_magic(runllm_items,item)
            #random.choice(list_of_monsters + list_of_npcs + list_of_rooms).add_magic(runllm_items,item)
            print (f'Adding riddle_magic {item} to {chosen.name}') 
    else:
        print ("No NPCs or riddle_magic_items")

def generating_room_disc_llm(): # actually to console since it is game set up
    print ("Generating room descriptions by llm")
    num_rooms = len(rooms)
    for number in range(1, num_rooms + 1):
        rooms[number].llM_generate_self_description()
    return 

def drawing_rooms_Diff(): # actually to console since it is game set up
    print ("Drawing rooms")
    num_rooms = len(rooms)
    for number in range(1, num_rooms + 1):
        rooms[number].drawing_self()
    return rooms

def connecting_rooms(): # this is actaully to console since it is game set up
    print ("Generating connections")
    num_rooms = len(rooms)
    row_size = int(num_rooms**0.5)
    for number in range(1, num_rooms+1):
        connect = ""
        while not connect:
            if number % row_size != 0 and "east" not in rooms[number].connected_rooms:   # Connect rooms east-west, but not for the last room in a row        
                if random.randint(0, 1):  # 50% chance to create a connection
                    rooms[number].add_connection("east", rooms[number + 1])
                    connect = "east"
            else:
                connect = "end"
            if number <= num_rooms-row_size and "south" not in rooms[number].connected_rooms:  # Connect rooms north-south, but not for the last row
                if random.randint(0, 1):  # 50% chance to create a connection
                    rooms[number].add_connection("south", rooms[number + row_size])
                    connect = str(connect) + "south"
            else:
                connect = "bottom"
    return 


class AdventureGame:
    def __init__(self, name ="", description ="", current_room=None):
        self.name = name
        self.description = description
        self.current_room = current_room
        self.items = []
        self.image = None

    def add_item(self, item):
        self.items.append(item)

    def get_item_names(self):
        return  ', '.join([item.name for item in self.items])
    
    def draw_and_save(self, prompt):
        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=25,
            guidance_scale= 7,
            decoder_guidance_scale=7,
            num_images_per_prompt=1,
                    ).images[0]
        self.image = image
        short_name = self.name[:15]
        timestamp = datetime.now().strftime("%m%d-%H%M")
        file_name = sanitize_filename(f"{short_name}_{timestamp}.png")
        image.save(os.path.join(image_dir, file_name))
        print (f"Image saved as {file_name}")
        return file_name
    
    def add_weapon(self, runllm_items = False, name = None):
        global weapons, copy_weapons
        if name == None:
            weapon = random.choice(weapons)
            weapons.remove(weapon) # so no duplicates
            if len(weapons) == 1:
                print ("Refreshing weapon list from copy_weapons")
                weapons = copy_weapons.copy()
        self.add_item(Weapon(name = weapon, damage = random.randint(10, 50), runllm = runllm_items))

    def add_armor(self, runllm_items = False, name = None):
        global armors, copy_armors
        if name == None:
            armor = random.choice(armors)
            armors.remove(armor)
            if len(armors) == 0:
                print ("Refreshing armor list from copy_armor")
                armors = copy_armors.copy()  
        self.add_item(Armor(name = armor, protection = random.randint(5, 25), runllm = runllm_items))

    def add_treasure(self, runllm_items = False, name = None): #because trophies can be added later
        global treasures, copy_treasures
        if name == None:
            treasure = random.choice(treasures)
            treasures.remove(treasure) # so no duplicates
            if len(treasures) == 0:
                print ("Refreshing treasure list from copy_treasures")
                treasures = copy_treasures.copy()
        else:
            treasure = name
        self.add_item(Treasure(name = treasure, value = random.randint(50, 1000), runllm=runllm_items))

    def add_magic(self, runllm_items = False, riddle_magic_items = None):
        global magic_items, copy_magic_items
        if riddle_magic_items == None:
            magic = random.choice(magic_items)
            magic_items.remove(magic) # so do duplicates
            if len(magic_items) == 0:
                print ("Refreshing magic list from copy_magic_items")
                magic_items = copy_magic_items.copy()
        else:
            magic = riddle_magic_items
        self.add_item(Magic(name = magic, healing = random.randint(20, 200), runllm = runllm_items))   

    def equip_self(self, llm_items = False):  #should make just like we do forrooms.
        weapon_count = generate_number(0,2,3,6) # 90% chance of 0, 1, or 2 weapons, 10% chance of 3, 4, or 5 weapons
        armor_count = generate_number(0,1,2,3) # 90% chance of 0, 1, armor, 10% chance of 2, 3 armor
        treasure_count = generate_number(0,2,3,6) # 90% chance of 0, 1, or 2 treasures, 10% chance of 3, 4, or 5 treasures
        magic_count = generate_number(0,2,3,6) # 90% chance of 0, 1, or 2 magic items, 10% chance of 3, 4, or 5 magic items
        for _ in range(weapon_count):
            self.add_weapon(llm_items)
        for _ in range(armor_count):
            self.add_armor(llm_items)
        for _ in range(treasure_count):
            self.add_treasure(llm_items)
        if isinstance(self, Room):          #the only magjic items that NPC or Monsters have are riddle_magic_items added latter
            for _ in range(magic_count):
                self.add_magic(llm_items)

    def drawing_self(self, style =None, victory = False):
        prompt=""
        if isinstance(self, Room):
            if art_style_g:
                prompt = (f"in {art_style_g} style.")
            if victory:
                prompt = victory
            else:
                prompt = prompt + self.description + "with weapons, treasure and magic items: " + self.get_item_names() + " and monsters: " + self.get_monster_names() 
                if self.npc:
                    if self.npc.image == None:
                        prompt = prompt + " and " + self.npc.description
        elif isinstance(self, Item):
            prompt = (f'{style} a {self.type} {self.description} in an adventue game')
        elif isinstance(self, LivingThing):
            prompt = (f'{style} a {self.description} in an adventue game')
        console = self.draw_and_save(prompt)
        return console
    
    def llM_generate_self_description(self): #Item descriptions should be 66 or few tokens since most art programs have that limit.
        if isinstance(self, Item):
            response = model(f"Very briefly describe this {self.type} called {self.name} in a cave based adventure game:?", max_tokens = 66 ,temperature = 0.3, repeat_penalty =1.5, stream = False)
            self.description = response ['choices'][0]['text']
        elif isinstance(self, LivingThing):
            response = model(f"Very briefly describe this {self.name} in a cave based adventure game:?", max_tokens = 66 ,temperature = 0.3, repeat_penalty =1.5, stream = False)
            self.description = response ['choices'][0]['text'] 
        elif isinstance(self, Room): #This is a full description from the short description that was previously generated.
            items = "items: " + self.get_item_names() + ", monsters: " + self.get_monster_names() 
            if self.npc:
                items = items + " and other character " + self.npc.description
            response = model (f"Expand on this description for an rooom in an adventure game: {self.description} adding these {items} into the description.", max_tokens = 256 ,temperature = 0.1, repeat_penalty =1.5, stream = False ) # Add item_text to encourage items
            self.full_description = response ['choices'][0]['text']
            print (self.full_description)
            return self.full_description
        print (self.description)
        return self.description

class Room(AdventureGame):
    def __init__(self, room_desc_riddle, number):
        super().__init__(name="Room " + str(number), description=room_desc_riddle["description"])
        self.room_desc_riddle = room_desc_riddle
        #self.description = room_desc_riddle["description"]
        self.full_description = room_desc_riddle["description"]
        self.riddles = room_desc_riddle["riddles"]
        self.hints = room_desc_riddle["hints"]
        self.riddle_magic_items = room_desc_riddle["magic_items"]
        self.riddle_action_words = room_desc_riddle["action_words"]
        self.number = number
        self.connected_rooms = {}
        self.monsters = [] 
        self.npcs = [] 
        self.npc = None 
        self.explored = False
        self.riddle_solved = False

    def add_connection(self, direction, room):
        self.connected_rooms[direction] = room
        if self not in room.connected_rooms.values():
            room.connected_rooms[self.opposite_direction(direction)] = self

    @staticmethod
    def opposite_direction(direction):
        return {"north": "south", "south": "north", "east": "west", "west": "east"}[direction]

    def get_monster_names(self):
        return ', '.join([monster.name for monster in self.monsters])

    def add_npc(self, llm_npc = False): # only in rooms
        global npcs, npc_descs, copy_npcs, copy_npc_descs
        npc = random.choice(npcs)
        npcs.remove(npc) # so do duplicates
        if len(npcs) == 0:
            npcs = copy_npcs.copy()
        npc_desc =random.choice(npc_descs)
        npc_descs.remove(npc_desc) # so no duplicates
        if len(npc_descs) == 0:
            print ("Refreshing npc_descs list from copy_npc_descs")
            npc_descs = copy_npc_descs.copy()
        self.npc = (NPC(name = npc, desc = npc_desc, current_room = self.number, llm_npc = llm_npc))

    def add_monster(self, runllm_mon = False): # only in rooms
        global monsters, copy_monsters
        monster = random.choice(monsters)
        monsters.remove(monster) # so no duplicates
        if len(monsters) == 0:
            print ("Refreshing monster list from copy_monsters")
            monsters = copy_monsters.copy()
        self.monsters.append(Monster(name = monster, health = random.randint(25, 200), damage = random.randint(10, 50), runllm_mon= runllm_mon))
    
    def populate_room(self, llm_mon = False, llm_npc = False):
        monster_count = generate_number(0,2,3,6) # 90% chance of 0, 1, or 2 monsters, 10% chance of 3, 4, or 5 monsters
        for _ in range(monster_count):
            self.add_monster(llm_mon)
        if random.randint(0, 4) == 0:  # 1 in 5 rooms has an NPC
            self.add_npc(llm_npc)
        return
        
    def inventory_room(self): #  Room inventory lists items - high level, monsters, and npc
        console = ""
        if isinstance(self, Room):
            if self.items:
                console = ("In the room you see: ")
                for item in self.items:
                    console = console + (f"{item.name}, ") # you don't know value until you get the items.    
                console = console[:-2] + (".\n")  
            if self.monsters:
                console = console + "There are monsters: "
                for monster in self.monsters:
                    console = console + (f"{monster.name} is here, it has {monster.health} health and can do {monster.damage} damage to you.\n")
            if self.npc:
                console = console + (f"{self.npc.name} is here, {self.npc.description}.\n")
        return console
    

class Item(AdventureGame):
    def __init__(self, name, type, runllm_item = False):
        super().__init__(name=name)
        self.type = type
        self.embeded_name = vector_model.encode(name)
        if runllm_item:
            self.llM_generate_self_description()
            self.drawing_self()
   
class Weapon(Item):
    def __init__(self, name, damage, runllm=False):
        super().__init__(name = name, type = "weapon", runllm_item = runllm)
        self.damage = damage
        if self.description == None:
            self.description = (f"weapon that can deal {self.damage} damage.")

class Armor(Item):
    def __init__(self, name, protection, runllm=False):
        super().__init__(name, type = "armor", runllm_item = runllm)
        self.protection = protection
        if self.description == None:
            self.description = (f"armor that can protect you from {self.protection} damage.")

class Treasure(Item):
    def __init__(self, name, value, runllm=False):
        super().__init__(name, type = "treasure", runllm_item = runllm)
        self.value = value
        if self.description == None:
            self.description = (f"treasure worth {self.value} shekel.")

class Magic(Item):
    def __init__(self, name, healing, runllm=False):
        super().__init__(name, type = "magic", runllm_item = runllm)
        self.healing = healing
        if self.description == None:
            self.description = (f"magic that can heal {self.healing}.")

class LivingThing(AdventureGame):
    def __init__(self, name = "jack", current_room = None, description = None, health = None):
        super().__init__(name=name, description=description, current_room = current_room, )
        self.health = health
        self.embeded_name = vector_model.encode(name) #could just put in adventer game class
        self.armors = []

    def add_item(self, item):
        self.items.append(item)

    def get_item_names(self):
        return ', '.join([item.name for item in self.items])

    def remove_item(self, item):
        self.items.remove(item)   

    def wear_armor(self, armor):
        self.remove_item(armor)
        self.armors.append(armor)

    def remove_armor(self, armor):
        self.armors.remove(armor)
        self.add_item(armor)    

    def move(self, direction):
        global rooms # since you are moving the NPC.
        console = ""
        if isinstance(self, NPC): # since NPC can move to a room with another NPC
            npc_new_room = rooms[self.current_room].connected_rooms[direction].number # new room number current room is just the room number of NPC
            if rooms[npc_new_room].npc: # If there is an NPC in the room
                npc_swap = rooms[npc_new_room].npc # Save the NPC in room you want to go to
                rooms[npc_new_room].npc = self # Move the current NPC to the new room
                rooms[self.current_room].npc = npc_swap # Swap the NPC in the current roomse
                console = console + (f"{self.name} went {direction} to room {npc_new_room}. {npc_swap.name} came into the room.\n")
            else:
                old_room = self.current_room
                self.current_room = npc_new_room
                rooms[npc_new_room].npc = self
                console = console + (f"{self.name} went {direction} to room {npc_new_room}.\n")
                rooms[old_room].npc = None #make sure he leaves the room :)
                print (console)
        else: # player current_room is a room class
            if direction in self.current_room.connected_rooms:
                self.current_room = self.current_room.connected_rooms[direction]
                console = (f"{self.name} moved {direction}.\n")
            else:
                console = (f"{self.name} can't go that way.\n")
        return console


    def inventory_living(self): #for living things details on all items # coulds update to do with room, and just hide some info
        response = ""
        if self.items:
            weapons = [item for item in self.items if isinstance(item, Weapon)]
            armors = [item for item in self.items if isinstance(item, Armor)]
            treasures = [item for item in self.items if isinstance(item, Treasure)]
            magic_items = [item for item in self.items if isinstance(item, Magic)]

            if not isinstance(self, Player):
                response = f"{self.name} has "
            if weapons:
                response += "Weapons: "
            for weapon in weapons:
                response += f"{weapon.name}, that can deal {weapon.damage} damage. "
            if armors:
                response += "Armors: "
            for armor in armors:
                response += f"{armor.name}, that can protect you from {armor.protection} damage. "
            if treasures:
                response += "Treasures: "
            for treasure in treasures:
                response += f"{treasure.name}, worth {treasure.value} shekel. "
            if magic_items:
                response += "Magic: "
            for magic in magic_items:
                response += f"{magic.name}, that can heal {magic.healing}. "
        else:
            response = f"{self.name} has nothing.\n"

        return response


class Player(LivingThing):
    def __init__(self, name = "Jack", description = None, current_room = None , runllm_diff = False, ):
        super().__init__(name = name, description= description, current_room = current_room) # this should fix the problem
        self.health = 300
        self.points = 0
        self.wealth = 0
        self.kills = 0
        self.equip_self()
        print ("Player current room from player class", self.current_room.number)
        if runllm_diff :
            self.llM_generate_self_description()
            self.drawing_self()
    
    def llm_describe_room(self): # calls llm for room to redescribe it usually after a change
        console =""
        console = console + self.current_room.llM_generate_self_description()
        return console
    
    def diff_draw_room(self): 
        console = self.current_room.drawing_self() # console is file name
        console = "New room image.\n"
        return console
    
    def health_up(self, health):
        if cheat_mode: # cheat mode is global
            self.health = int (health)
            console = (f"Health set to {health}.\n")
        else:
            console = ("You can't cheat without activating cheat mode with magic word.\n")
        return console

    def magic_connections(self, room_number=None):
        if room_number == None:
            room_number = self.current_room.number
        console = riddle_create_connections(room_number)
        return console

    def _find_item(self, item_name, items):
        for item in items:
            if fuzz.ratio(item.name.lower(), item_name.lower()) > 60 or item_name.lower() in item.name.lower():
                return item
        return None

    def _describe_single_item(self, item, addvalue = None): #full description
        console = ""
        console = (f"This is {item.name} a {item.type}, ") #full description
        if addvalue == "study":
            console = console + (f"it is a {item.description}.\n")
        if isinstance(item, Weapon):
            console = console + (f"it can deal {item.damage} damage.\n")
        elif isinstance(item, Armor):
            console = console + (f"it can protect you from {item.protection} damage.\n")
        elif isinstance(item, Treasure):
            console = console + (f"it is worth {item.value}.\n")
            if addvalue == "positive":
                self.wealth += item.value
                self.points += 5
            elif addvalue == "negative":
                self.wealth -= item.value
                self.points -= 5
        elif isinstance(item, Magic):
            console = console + (f"it can heal {item.healing}.\n")
        return console

    def study(self, item_name): # return full description of item and item.image
        item = self._find_item(item_name, self.current_room.items)
        item_images = []
        if item:
            console = self._describe_single_item(item, addvalue = "study")
            if item.image:
                item_images.append(item.image)    
                return console, item.image
            else:
                return console
        monster = self._find_character(item_name, self.current_room.monsters)
        if monster: 
            console = (f"This {monster.name} has {monster.health} health and can do {monster.damage} damage to you. {monster.description}\n")
        else:
            console = (f'{item_name} not in the room.\n' )
        return console
    
    def take(self, item_names):
        if isinstance(item_names, str):
            item_names = [item_names]  # convert single string to list
        consoles = []
        item_images = []
        for item_name in item_names:
            item = self._find_item(item_name, self.current_room.items)
            if item:
                self.items.append(item)
                console = self._describe_single_item(item, "positive")
                consoles.append(console)
                self.current_room.items.remove(item)
                if item.image:
                    item_images.append (item.image)
            else:
                console = ("{} isn't here.\n".format(item_name))
                consoles.append(console)
        return "\n".join(consoles), item_images

    def leave(self, item_names):
        if isinstance(item_names, str):
            item_names = [item_names]  # convert single string to list
        consoles = []
        item_images = []
        for item_name in item_names:
            item = self._find_item(item_name, self.items)
            if item:
                self.current_room.items.append(item)
                console = self._describe_single_item(item, "negative")
                consoles.append(console)
                self.items.remove(item)
                if item.image:
                    item_images.append(item.image)
            else:
                console = ("{} isn't in your inventory.\n".format(item_name))
                consoles.append(console)
        return "\n".join(consoles), item_images
    

    def solve_puzzle(self, item1_name, item2_name=None): # update to be agnostic to order
        console = ""
        print (item1_name, item2_name)
        if self.current_room.riddle_solved:
            return "You already solved the riddle!\n"
        item1 = self._find_item(item1_name, self.items)
        
        if not isinstance(item1, Magic):
            return "You can't use that item to solve the riddle.\n"
        if item2_name:
            item2 = self._find_item(item2_name, self.items)

            if item2 and not isinstance(item2, Magic):
                print ("item2 not magic")
                item2 = None
        else:
            item2 = None

        riddle_items = self.current_room.riddle_magic_items
        print (riddle_items, item1.name, item2_name)
        if item1.name in riddle_items and len(riddle_items) == 1 or \
        item1.name in riddle_items and item2 is not None and item2.name in riddle_items:
            self.current_room.riddle_solved = True
        else:
            return "That didn't solve the riddle.\n"
        if self.current_room.riddle_solved:
            console = f"{item1.name} evaporates into thin air!\n"
            self.items.remove(item1)
            if isinstance(item2, Magic):
                console += f"{item2.name} evaporates into thin air!\n"
                self.items.remove(item2)
            console += f"You solved the riddle: {self.current_room.riddles[0]}!\n"
            console += riddle_create_connections(self.current_room.number)
            console += f"** Secret Unlocked: Say {magicword} to open passages.\n"
            self.points += item1.healing * 10
            self.health += item1.healing * 3
        return console
        
    def use(self, item_name):
        item_images = []
        item = self._find_item(item_name, self.items)
        if item:
            if isinstance(item, Magic):
                self.health += item.healing
                self.points -= 5
                self.items.remove(item) # you need to remove it after you check for image
                console = (f"You used the {item.name} and restored {item.healing} health!")
                if item.image:
                    item_images.append(item.image)
                    
                    return console, item_images
                else:
                    
                    return console
                
            else:
                console = ("You can't use that item.")
        else:
            console = ("You don't have that item.")     
        return console
      
    def put_on_armor(self, armor_name):
        armor = self._find_item(armor_name, self.items)
        if armor:
            self.wear_armor(armor)
            console = (f"You put on the {armor.name}.")
        else:
            console = ("You don't have that armor.")
        return console
    
    def take_off_armor(self, armor_name):
        armor = self._find_item(armor_name, self.armors)
        if armor:
            self.remove_armor(armor)
            console = (f"You took off the {armor.name}.")
        else:
            console = ("You don't have that armor on.")
        return console
    
    def _find_character(self, name, characters):
        for character in characters:
            if fuzz.ratio(character.name.lower(), name.lower()) > 60 or name.lower() in character.name.lower():
                return character
        return None
    
    def about_to_npc(self, input_text= "what do you want to know about this world"):
        console=""
        npc = self.current_room.npc
        if npc:
            console = npc.llM_generate_self_info(dialogue = input_text)
        else:
            console = "No one in room to talk to.\n"
        return console # do you want to use second return for image?

    def talk_to_npc(self, input_text= "what do you have to trade"):
        console=""
        npc = self.current_room.npc
        my_stuff = self.get_item_names()
        if npc:
            console = npc.llM_generate_self_dialogue(dialogue = input_text, my_stuff = my_stuff)
        else:
            console = "No one in room to talk to.\n"
        return console # do you want to use second return for image?

    def trade_with_npc(self, my_item_name=None, their_item_name = None): # updated with response
            console = ""
            npc = self.current_room.npc
            if npc:
                my_item = self._find_item(my_item_name, self.items)
                their_item = self._find_item(their_item_name, npc.items)
                if my_item and their_item:
                    self.items.remove(my_item)
                    console = self._describe_single_item(my_item, "negative") # just using to adjust wealth & points
                    npc.items.remove(their_item)
                    self.items.append(their_item)
                    console = console + self._describe_single_item(their_item, "positive")
                    npc.items.append(my_item)
                    console = console + (f'Thank you for the trade, you now have the {their_item_name} and I have the {my_item_name}.\n')
                else:
                    console = ("Don't try to cheat me!\n")      
            else:
                console = ("There is no one here to trade with.\n")
            return console

    def attack(self, monster_name, weapon_name):  
        global you_started_it
        monster = self._find_character(monster_name, self.current_room.monsters)
        if monster:
            weapon = self._find_item(weapon_name, self.items)
            if weapon:
                you_started_it = True
                monster.health -= weapon.damage
                if monster.health <= 0:
                    console = (f"!! You killed the {monster.name} it dropped {monster.get_item_names()}\n") # eventually use monster classes
                    console = console + monster.description + "\n"
                    self.current_room.items.extend(monster.items)
                    self.current_room.monsters.remove(monster)
                    self.kills += 1
                    self.points += 10
                else:
                    console = (f"!! You hit the {monster.name} with {weapon.name} at strength {weapon.damage}!  Monster health is now {monster.health}\n")
            else:
                console = ("You don't have that weapon.\n")
        else:
            console = ("That monster isn't here.\n")
        return console
    
    
class NPC(LivingThing):
    def __init__(self, name, desc, current_room, llm_npc = False):
        super().__init__(name = name, description = desc, current_room = current_room)
        self.health = 100
        self.equip_self()

        if llm_npc:
            self.drawing_self()
            self.llM_generate_self_description()

    #could use our get entities to get list of items to trade
    def llM_generate_self_dialogue(self, dialogue = "what do you have to trade", my_stuff = None):
        response = model(f"You are {self.description} in an adventure game asked {dialogue}, answer in character. You have to trade {self.get_item_names()}, and I have {my_stuff}", max_tokens = 128 ,temperature = 0.7, repeat_penalty =1.5, stream = False)
        response = response['choices'][0]['text']
        response = (f"{self.name} says: {response}\n")
        response_embeddings = vector_model.encode(response)
        distances = util.pytorch_cos_sim(response_embeddings,dict_action_embeddings["trade"])
        max_score, max_index = torch.max(distances, dim=1)
        entity = dict_actions["trade"][max_index.item()]
        print (f"Max score for {entity} is {max_score.item()}")
        if max_score > 0.25:
            response = response + self.inventory_living() + "\n"
        return response
    
    def llM_generate_self_info(self, dialogue = "Which rooms have the riddels"):
        riddle_answers = riddles_and_magic()
        response = model(f"You are a helpful NPC in an adventrue game, answer {dialogue} based on the following paragraphs: {riddle_answers}", max_tokens = 500 ,temperature = 0.01, repeat_penalty =1.5, stream = False)
        response = response['choices'][0]['text']
        response = (f"{self.name} says: {response}\n")
        return response


class Monster(LivingThing):
    def __init__(self, name, health, damage, runllm_mon = False):
        super().__init__(name = name, health = health)
        self.damage = damage 

        if runllm_mon:
            self.llM_generate_self_description()
            self.drawing_self()

    def monster_attack(self, player): # eventually move this to the monster class
        #need to fix so only one amour is used
        protection = 0
        if player.armors:
            for armor in player.armors:
                protection += armor.protection
        protected_damage = max(self.damage - protection, 0)
        if protected_damage == 0:
            console = (f"** The {self.name} missed you. !\n")
        else:
            player.health = player.health - protected_damage
            console = (f"** The {self.name} hit you for {protected_damage} damage! Your health is now {player.health}\n")
        return console        
    
def draw_map_png(cheat_mode=False):
    num_rooms = len(rooms)
    num_rooms_side = int(num_rooms**0.5)
    room_size = 60  # Size of one room square in pixels
    connection_size = 8  # Width of the connection lines in pixels
    room_size = 60  # Offset of the west connection line from the center of the room in pixels
    room_size = 60  # Offset of the north connection line from the center of the room in pixels
    w, h = 10,10 # adjustment for room numbers
    
    # Calculate the total size of the image
    image_size = ((num_rooms_side) * room_size, (num_rooms_side) * room_size)
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Load a default font
    
    for number, room in rooms.items():
        x = ((number - 1) % num_rooms_side) * room_size
        y = ((number - 1) // num_rooms_side) * room_size
        if room.explored or cheat_mode:
            if room.image:
                imagetopaste = room.image.resize((room_size,room_size), Image.LANCZOS)
                image.paste(imagetopaste, (x, y))
        #draw.rectangle([x, y, x + room_size, y + room_size], outline="black", )
        
    # Draw connections (if rooms are explored or in cheat mode)
    for number, room in rooms.items():
        x = ((number - 1) % num_rooms_side) * room_size
        y = ((number - 1) // num_rooms_side) * room_size
        if room.explored or cheat_mode:
            for direction, connected_room in room.connected_rooms.items():
                if direction == "east":
                    draw.line([(x + room_size/1.2, y + room_size / 2 ), 
                               (x + room_size*1.2, y + room_size / 2 )], 
                              fill="green", width=connection_size)
                if direction == "south":
                    draw.line([(x + room_size/2, y + room_size / 1.2 ), 
                               (x + room_size/2, y + room_size * 1.2 )], 
                              fill="blue", width=connection_size)
                if direction == "west":
                    draw.line([(x - (room_size / 1.2 - room_size), y + room_size/2), 
                               (x - (room_size * 1.2 - room_size), y + room_size/2)], 
                              fill="pink", width=connection_size)
                if direction == "north":
                    draw.line([(x + room_size / 2 , y + room_size / 1.2 - room_size), 
                               (x + room_size / 2 , y + room_size * 1.2 - room_size)], 
                              fill="orange", width=connection_size)
        draw.text((x + (room_size - w) / 2, y + (room_size - h) / 2), str(number), fill="black", font=font)
    #image.show()
    image.save(f"map.png")
    return image

def riddle_create_connections(number):
    num_rooms = len(rooms)
    number = int(number)
    if number > num_rooms or number < 1:
        return "No such room."
    row_size = int(num_rooms**0.5)
    connect = ""
    if number % row_size != 0 and "east" not in rooms[number].connected_rooms:   # Connect rooms east-west, but not for the last room in a row   # Connect rooms east-west, but not for the last room in a row        
        rooms[number].add_connection("east", rooms[number + 1])
        connect = "east, "
    if number > row_size and "north" not in rooms[number].connected_rooms:  # Connect rooms north-south, but not for the first row
        rooms[number].add_connection("north", rooms[number - row_size])
        connect = str(connect) + "north, "
    if number >1 and number % row_size != 1 and "west" not in rooms[number].connected_rooms:  # Connect rooms west-east, but not for the first column
        rooms[number].add_connection("west", rooms[number - 1])
        connect = str(connect) + "west, "
    if number <= num_rooms-row_size and "south" not in rooms[number].connected_rooms:  # Connect rooms north-south, but not for the last row
        rooms[number].add_connection("south", rooms[number + row_size])
        connect = str(connect) + "south. "
    console = (f"Magic connection for {str(number)}: {connect[:-2]}.\n")
    return console

def cheat_code(cheat_code=None): #called by xyzzy
    global cheat_mode
    if cheat_code == "xyzzy" or cheat_code == magicword:
        cheat_mode = True
    return "Cheat mode enabled. To create connections: " + magicword + " <room number>. To increase health: Health <quantity>.\n"
    
def get_help():
    console = help
    return console

def initalize_game_varables():
    global rooms, npc_introduced, you_started_it, cheat_mode, old_room, trophies, active_game, art_style_g, dict_actions, dict_action_embeddings, magicword
    npc_introduced = False
    you_started_it = False
    cheat_mode = False
    old_room = None
    trophies = []
    active_game = False
    art_style_g = ""
    rooms = {} # dictionary of rooms room number is they key, room objecters
    magicword = random.choice(["abbracadabra", "alakazam", "hocuspocus", "opensesame", "shazam", "presto", "ivy", "blackjack","titi", "nikki", "versace"])
    dict_actions["magicword"] = [magicword]
    dict_action_embeddings["magicword"] = [vector_model.encode(magicword)]


help = """
Commands:
go <direction> - north, south, east, or west
take <item> - take an item from the room
leave <item> - leave an item in the room
use <item> - use an item in your inventory
study <item> - study an item in the room
attack <monster> with <weapon> 
trade <item> for <item> - trade inventory for NPC item
talk <what you want to say to the> - talk to NPC 
details - update room description
draw - update room image 
solve <item> and <item> - solve riddle items from inventory
Hint - Monsters have items to solve riddles
help - show this help message
"""

active_game = False
diff_dict = {"cheat": 1, "easy": 1, "medium": 2, "hard": 3}
#npc_name = "Zac" # place holder to remove from commands
playerOne = None
rooms = {} # dictionary of rooms room number is they key, room objecters
list_of_saved_games = saved_files()
list_of_llms = saved_llms()
list_of_diffusers = saved_diffusers()


def find_similar_action(user_input):
    global dict_action_embeddings, dict_actions
    user_input_embedding = vector_model.encode(user_input)
    best_actions = [None, None]
    best_scores = [0, 0]
    extracted_actions = [{}, {}]
    for action_type, embeddings in dict_action_embeddings.items():
        distances = util.pytorch_cos_sim(user_input_embedding, embeddings)
        max_score, max_index = torch.max(distances, dim=1)
        action_list = dict_actions[action_type]
        if max_score > 0.3:
            print(f"Best: {action_type} {max_score.item():.2f} Extracted Action {action_list[max_index.item()]}")
        if max_score > best_scores[0] and max_score > 0.2:  # Threshold for considering a match
            best_actions[1] = best_actions[0]
            best_scores[1] = best_scores[0]
            extracted_actions[1] = extracted_actions[0]
            best_actions[0] = action_type
            best_scores[0] = max_score.item()
            extracted_actions[0] = {action_type: action_list[max_index.item()]}
        elif max_score > best_scores[1] and max_score > 0.2:
            best_actions[1] = action_type
            best_scores[1] = max_score.item()
            extracted_actions[1] = {action_type: action_list[max_index.item()]}
    print (f'Best Two Actions: {best_actions} {best_scores[0]:.2f} {best_scores[1]:.2f}, Extracted Actions: {extracted_actions} \n')
    return best_actions

#For update to get all the matches
def extract_entities(user_input, dict_entity_embeddings, dict_entities):
    user_input_embedding = vector_model.encode(user_input)
    extracted_entities = {}
    for entity_type, embeddings in dict_entity_embeddings.items():
        if embeddings:
            distances = util.pytorch_cos_sim(user_input_embedding, embeddings)
            # Create a mask for distances over the threshold
            mask = distances > 0.35 # to not grab the wrong items. 
            # Apply the mask to get the indices of distances over the threshold
            indices = torch.where(mask)[1]
            # Get the corresponding distances
            valid_distances = distances[0][indices]
            # Sort the indices by the valid distances in descending order
            sorted_indices = indices[torch.argsort(valid_distances, descending=True)]
            # Use the sorted indices to get the entities
            entity_list = dict_entities[entity_type]
            sorted_entities = [entity_list[i] for i in sorted_indices]
            sorted_scores = ["{:.2f}".format(valid_distances[i].item()) for i in torch.argsort(valid_distances, descending=True)]
            if sorted_entities:
                extracted_entities[entity_type] = sorted_entities
                print(f"Entity Type: {entity_type}, Entities: {sorted_entities}, Scores: {sorted_scores}")
    return extracted_entities


def execute_action(chosen_action, entities_by_type, command):
    chosen_action = chosen_action[0] if chosen_action else None
    chosen_action_2nd = chosen_action[1] if chosen_action[1] else None
    list_entities_by_type = entities_by_type
    # This is a dictionary comprehension. It iterates over the items in the dictionary 'list_entities_by_type'.
    # For each item, it checks if the value (v) is not empty. If it's not empty, it adds an entry to the new dictionary.
    # The key of the new entry is the key from the original dictionary (k), and the value is the first element of the value from the original dictionary (v[0]).
    entities_by_type = {k: v[0] for k, v in list_entities_by_type.items() if v}  
    # This is another dictionary comprehension. It also iterates over the items in the dictionary 'list_entities_by_type'.
    # For each item, it checks if the value (v) is not empty. If it's not empty, it adds an entry to the new dictionary.
    # The key of the new entry is the key from the original dictionary (k).
    # The value of the new entry is the second element of the value from the original dictionary (v[1]) if the value has more than one element. If it doesn't, the value is None.
    entities_by_type_2nd = {k: v[1] if len(v) > 1 else None for k, v in list_entities_by_type.items() if v}  # Only using the second entity if it exists

    action_map = {
        "go": lambda: playerOne.move(entities_by_type.get("directions")) if entities_by_type.get("directions") is not None else "Directions not specified.",
        "attack": lambda: playerOne.attack(entities_by_type.get("monsters"), entities_by_type.get("weapons")) if entities_by_type.get("monsters") is not None and entities_by_type.get("weapons") is not None else "Monsters or weapons not specified.",
        "trade": lambda: playerOne.trade_with_npc(entities_by_type.get("my_items"), entities_by_type.get("npc_items")) if entities_by_type.get("my_items") is not None and entities_by_type.get("npc_items") is not None else \
              playerOne.talk_to_npc(command), #hack since often you have words that trigger trading, in your questions, but you don't know what to trade
        "talk": lambda: playerOne.talk_to_npc(command) if command is not None else "What to say not specified.",
        "about": lambda: playerOne.about_to_npc(command) if command is not None else "What to say not specified.",
        "use": lambda: playerOne.use(entities_by_type.get("my_magic")) if entities_by_type.get("my_magic") is not None else "Magic items not specified.",
        "wear": lambda: playerOne.put_on_armor(entities_by_type.get("my_armor")) if entities_by_type.get("my_armor") is not None else "Armor not specified.",
        "remove": lambda: playerOne.take_off_armor(entities_by_type.get("wear_armor")) if entities_by_type.get("wear_armor") is not None else "Armor not on.",
        "take": lambda: playerOne.take(list_entities_by_type.get("room_items")) if list_entities_by_type.get("room_items") is not None else "Room items not specified.", #using list!
        "leave": lambda: playerOne.leave(list_entities_by_type.get("my_items")) if list_entities_by_type.get("my_items") is not None else "My items not specified.",
        "solve": lambda: playerOne.solve_puzzle(entities_by_type.get("my_magic"),entities_by_type_2nd.get("my_magic")) if entities_by_type.get("my_magic") is not None else "My magic items not specified.", #also 2nd item
        "study": lambda: playerOne.study(entities_by_type.get("room_items")) if entities_by_type.get("room_items") is not None else "Room items not specified.",
        "draw": lambda: playerOne.diff_draw_room(),
        "describe": lambda: playerOne.llm_describe_room(),
        "health": lambda: playerOne.health_up(int(re.findall('\d+', command)[0])) if re.findall('\d+', command) else "No health number specified.",
        "magicword": lambda: playerOne.magic_connections(int(re.findall('\d+', command)[0])) if re.findall('\d+', command) else "No room number specified.",
        "help": lambda: get_help(),
        "cheat": lambda: cheat_code(cheat_code= command)
    }
    return action_map.get(chosen_action, lambda: "I don't understand that command.")()

def embedings_for_entities_updated():
    room_items = [item.name for item in playerOne.current_room.items if isinstance(item, Item)] if playerOne.current_room.items else []
    room_items_embeded = [item.embeded_name for item in playerOne.current_room.items if isinstance(item, Item)] if playerOne.current_room.items else []
    room_monsters = [monster.name for monster in playerOne.current_room.monsters] if playerOne.current_room.monsters else []
    room_monsters_embeded = [monster.embeded_name for monster in playerOne.current_room.monsters] if playerOne.current_room.monsters else []
    my_weapons = [item.name for item in playerOne.items if isinstance(item, Weapon)] if playerOne.items else []
    my_weapons_embeded = [item.embeded_name for item in playerOne.items if isinstance(item, Weapon)] if playerOne.items else []
    my_armor = [item.name for item in playerOne.items if isinstance(item, Armor)] if playerOne.items else []
    my_armor_embeded = [item.embeded_name for item in playerOne.items if isinstance(item, Armor)] if playerOne.items else []
    my_magic = [item.name for item in playerOne.items if isinstance(item, Magic)] if playerOne.items else []
    my_magic_embeded = [item.embeded_name for item in playerOne.items if isinstance(item, Magic)] if playerOne.items else []
    my_items = [item.name for item in playerOne.items if isinstance(item, Item)] if playerOne.items else []
    my_items_embeded = [item.embeded_name for item in playerOne.items if isinstance(item, Item)] if playerOne.items else []
    npc_items  = [item.name for item in playerOne.current_room.npc.items if isinstance(item, Item)] if playerOne.current_room.npc and playerOne.current_room.npc.items else []
    npc_items_embeded  = [item.embeded_name for item in playerOne.current_room.npc.items if isinstance(item, Item)] if playerOne.current_room.npc and playerOne.current_room.npc.items else []
    npc_name = [playerOne.current_room.npc.name] if playerOne.current_room.npc else []
    npc_name_embeded = [playerOne.current_room.npc.embeded_name] if playerOne.current_room.npc else []
    wearing_armor = [item.name for item in playerOne.armors if isinstance(item, Armor)] if playerOne.armors else []
    wearing_armor_embeded = [item.embeded_name for item in playerOne.armors if isinstance(item, Armor)] if playerOne.armors else []
    
    dict_entities = {
        "my_magic": my_magic, 
        "weapons": my_weapons, 
        "my_armor": my_armor,
        "monsters": room_monsters, 
        "npcs": npc_name, 
        "room_items": room_items, 
        "my_items": my_items, 
        "npc_items": npc_items, 
        "wear_armor": wearing_armor,
        "directions": directions_g, 
        "values": values_g
    }
    dict_entity_embeddings = {
        "my_magic": my_magic_embeded, 
        "weapons": my_weapons_embeded,
        "my_armor": my_armor_embeded, 
        "monsters": room_monsters_embeded, 
        "npcs": npc_name_embeded,
        "room_items": room_items_embeded, 
        "my_items": my_items_embeded, 
        "npc_items": npc_items_embeded, 
        "wear_armor": wearing_armor_embeded,
        "directions": directions_embedded, 
        "values": values_embedded
    }
    return dict_entities, dict_entity_embeddings

def riddles_and_magic(): 
    room_riddles = ""
    for room_i in rooms.values(): 
        if room_i.riddles:  
            room_riddles += generate_riddle_info(room_i)
    print (room_riddles)
    return room_riddles

def generate_riddle_info(room_i):
    room_riddles = ""
    room_riddles = room_riddles + f"\nRoom Number {room_i.number} has this Riddle:\n{room_i.riddles[0]} \n"
    room_riddles = room_riddles+(f'These items are Needed to solve the riddle: {", ".join(map(str, room_i.riddle_magic_items))}. \n')    
    room_riddles = room_riddles + (f"Hint: {room_i.hints[0]} \n")
    if len(room_i.hints) > 1:
        room_riddles = room_riddles + (f"Hint 2: {room_i.hints[1]} \n")

    which_room_is_it = [(room.number, item.name) for room in rooms.values() for item in room.items if item.name in room_i.riddle_magic_items]
    which_npc_is_it = [(room.number, room.npc.name, item.name) for room in rooms.values() if room.npc for item in room.npc.items if item.name in room_i.riddle_magic_items]
    which_monster_has_it = [(room.number, monster.name, item.name) for room in rooms.values() for monster in room.monsters for item in monster.items if item.name in room_i.riddle_magic_items]
    if which_room_is_it:
        for room_number, item_name in which_room_is_it:
            room_riddles += f'The item "{item_name}" can be found in Room Number {room_number}.\n'
    if which_npc_is_it:
        for room_number, npc_name, item_name in which_npc_is_it:
            room_riddles += f'The item "{item_name}" can be found with the NPC "{npc_name}" in Room Number {room_number}.\n'
    if which_monster_has_it:
        for room_number, monster_name, item_name in which_monster_has_it:
            room_riddles += f'The item "{item_name}" can be found with the Monster "{monster_name}" in Room Number {room_number}.\n'   
    return room_riddles

# This is a helper function that checks if a trophy is already in the list.
# If not, it checks the rooms for a specific condition.
# If the condition is met, it adds the trophy to the list and updates the console.
def check_trophy(trophy, condition, victory_message, victory_action):
    global trophies
    # Check if the trophy is not already in the list
    player_image = None
    trophie_images = []
    console = ""
    if trophy not in trophies:
        all_conditions_met = True
        # Iterate over all rooms
        for room in rooms.values():
            # The condition function is called here with the room as argument.
            # This function is a lambda function that was passed as an argument to check_trophy.
            # It checks a specific condition in the room (e.g., if there are any treasures, monsters, riddles, or if the room is explored).
            if condition(room):
                all_conditions_met = False
        # If all conditions are met (i.e., there are no more treasures, monsters, riddles, or unexplored rooms), the player wins the trophy
        if all_conditions_met:
            console = console + (f"Congratulations, {victory_message}!\n")
            playerOne.drawing_self(victory_action)
            player_image = playerOne.image
            playerOne.current_room.add_treasure(False, f"{trophy.capitalize()}-Trophie")
            temp_console, trophie_images  = playerOne.take(f"{trophy.capitalize()}-Trophie")  # returns the console and the image of the trophy
            trophies.append(trophy)
            print (f"trophies: {trophies}")
            console = console +temp_console + f"Your trophies: {', '.join(trophies)}\n"
    
    player_image = [player_image] if player_image is not None else []
    #trophiie images are a list since take returns lists
    return console, player_image, trophie_images


def talk_to_functions(command, name_of_game, difficulty, game_state, new_game_name, number_of_rooms, llm_diff, llm_mon, 
                      llm_item,name_of_llm, name_of_diffuser, art_style):
    global playerOne, rooms, active_game, list_of_saved_games, trophies, difficulty_g, cheat_mode, npc_introduced, you_started_it,  old_room, art_style_g
    
    room_image = []
    map_image = None
    npc_image = []
    item_images = []
    monster_image = []
    player_image = []
    console = ""
    your_status = ""
    rooms_visted = ""
    room_description = ""
    room_long_description = ""
    room_riddles = ""
    room_contents = ""
    inventory1 = ""
    art_style_g = art_style # global variable for art_style in classes
    game_state_selector = "Play-Game"

    if not difficulty:
        difficulty = "easy"
    if not command:
        command = " "
    difficulty = difficulty.lower()
    difficulty_g = difficulty # global variable for difficulty 
    
    if  game_state == "New-Game" or game_state == "Load-Game":
        embedings_for_actions()
        initalize_game_varables()
        load_adventure_data()
        set_up_llm_diffuser(diff_name=name_of_diffuser, llm_name=name_of_llm)

    if game_state == "New-Game":
        console = new_game(num_rooms = number_of_rooms, runllm_diff= llm_diff, runllm_mon =llm_mon, runllm_item = llm_item, name_of_game = new_game_name)
        console = (f"New game: {console}.\n")
        active_game = True
        game_state_selector = gr.Radio(value="Play-Game")

    elif game_state == "Load-Game":
        rooms, playerOne, old_room=load_game_state(name_of_game)
        console = (f"Loaded game: {name_of_game}.\n")
        active_game = True
        game_state_selector = gr.Radio(value ="Play-Game")
        
    elif game_state == "Save-Game":
        name_of_game = save_game_state(new_game_name)
        console = (f"Saved game: {name_of_game}. Select Play-Game to continue playing.\n")
        game_state_selector = gr.Radio(value="Play-Game")
        list_of_saved_games = saved_files()

    elif game_state == "Play-Game" and active_game:
        if old_room != playerOne.current_room:
            you_started_it = False
            npc_introduced = False
        dict_entities, dict_entity_embeddings = embedings_for_entities_updated()
        print ("Command:",command) 
        chosen_action = find_similar_action(command)
        print ("Chosen Action:", chosen_action) # top  actions
        entities_by_type = extract_entities(command, dict_entity_embeddings, dict_entities)
        print ("Entities by type:", entities_by_type) # top  matches for each entity type
        result = execute_action(chosen_action, entities_by_type, command)
        if len(result) == 2:
            result, item_images = result
            console = console + result
        else:
            console = console + result
            item_images = []
    # return if no active game
    else: 
        print ("Returning from talk_to_functions early", room_image, map_image)
        return (room_image, map_image, 
        room_description.strip()+"\n"+room_riddles.strip(),
        room_contents.strip(), console.strip(), your_status,
        rooms_visted.strip(), inventory1.strip(),
        room_long_description, game_state_selector)

    #Room content
    room_long_description = playerOne.current_room.full_description    
    room_long_description = room_long_description.strip() 
    if difficulty in ["cheat", "easy", "medium"]:
        room_contents = room_contents + playerOne.current_room.inventory_room()
        room_long_description = gr.Textbox(visible=True, value = room_long_description) # togle back on.
    #Hard mode they need to figure out content from description
    elif playerOne.current_room.full_description != playerOne.current_room.description: 
        room_contents = room_contents + playerOne.current_room.full_description # so you figure out contents yourself
        room_long_description = gr.Textbox(visible=False)
    # Update action of monsters
    if playerOne.current_room.monsters:
        monsters = playerOne.current_room.monsters
        monster_content = ""
        for monster in monsters:
            attack_probability = random.randint(0, 9000/diff_dict[difficulty]) # like difficulty dynamic
            if attack_probability < playerOne.wealth or you_started_it: #if you attack anything it wakes monsterup 
                console = console + monster.monster_attack(playerOne) #moved to monster not player :) 
            monster_content = monster_content + monster.inventory_living()
            if monster.image:
                monster_image.append(monster.image)
        if difficulty in ["cheat", "easy"] and monsters:
            room_contents = room_contents + "Monster posessions: " + monster_content
    
    # Update action of NPC using NPC class
    if playerOne.current_room.npc: 
        npc = playerOne.current_room.npc
        if random.randint(0, 9) == 0 and list(playerOne.current_room.connected_rooms.keys()): #Does NPC move? #2 for testing
            direction = random.choice(list(playerOne.current_room.connected_rooms.keys()))
            console = console + npc.move(direction) #move the npc
        elif not npc_introduced:
            if random.randint(0, 3) == 0:
                console = console + npc.llM_generate_self_dialogue("You hear nothing from the visitor so you introduce yourself") ## this needs to be fixed, since it is not a string but instead writes to the console
                npc_introduced = True
        if difficulty in ["cheat", "easy"]: #NPC Inventory to help
            room_contents = room_contents + (f'NPC Inventory: {npc.inventory_living()}') 
        if npc.image:
            npc_image.append(npc.image)
            
    #Riddles
    if playerOne.current_room.riddles and playerOne.current_room.riddle_solved == False:
        room_riddles = (f"There is a riddle here: {playerOne.current_room.riddles[0]}.\n")
        if difficulty in ["cheat", "easy"]:
            room_riddles = room_riddles + (f"Hint: {playerOne.current_room.hints[0]} \n")
            if len(playerOne.current_room.hints) > 1:
                room_riddles = room_riddles + (f"Hint 2: {playerOne.current_room.hints[1]} \n")
        if cheat_mode and difficulty == "cheat":
            room_riddles = "Cheat: " + generate_riddle_info(playerOne.current_room)
   
    # Check for winning the game or death 
    # The lambda functions are passed as arguments to the check_trophy function.
    # They are not executed here, but inside the check_trophy function.
    # Return trophy images and player images as lists so don't have to check if None.
    console_temp, player_image_temp, item_images_temp = check_trophy("treasure", lambda room: room.items and any(isinstance(item, Treasure) for item in room.items), "you got all the treasures", "celebrating finding treasure")
    console += console_temp
    player_image += player_image_temp
    item_images += item_images_temp

    console_temp, player_image_temp, item_images_temp = check_trophy("monster", lambda room: room.monsters, "you killed all the monsters", "celebrating killing monsters")
    console += console_temp
    player_image += player_image_temp
    item_images += item_images_temp

    console_temp, player_image_temp, item_images_temp = check_trophy("riddle", lambda room: room.riddles and not room.riddle_solved, "you won the game all the riddles solved", "Celebration solving riddles")
    console += console_temp
    player_image += player_image_temp
    item_images += item_images_temp

    console_temp, player_image_temp, item_images_temp = check_trophy("explorer", lambda room: not room.explored, "you explored all the rooms", "Celebration succesfully exploring")
    console += console_temp
    player_image += player_image_temp
    item_images += item_images_temp 
    print (console)

    if playerOne.health <= 0:
        console = console + (f'Health: {int(playerOne.health)}  Wealth: {playerOne.wealth}  Points: {playerOne.points}  Kills: {playerOne.kills}\n')
        console = console + ("You died. Game over.\n")
        victory = ("Died, killed by a monsters")
        playerOne.drawing_self(victory)
        if playerOne.image:
            player_image.append(playerOne.image)
        console = console + (f"Your trophies: {', '.join(trophies)}\n")
        game_state_selector = gr.Radio(value="Load-Game")

    #update the image, description of the room, health, wealth, points, kills, map, inventory
    try:
        if playerOne.current_room.image:
            room_image.append(playerOne.current_room.image)
    except:
        room_image = []
    #Room description
    room_description =  room_description +(f'Room {playerOne.current_room.number}: {playerOne.current_room.description}\n')
    if not playerOne.current_room.explored:
            room_description = room_description + "You haven't been in this room before.\n"
            playerOne.current_room.explored = True # must set before draw_map or seeing if all rooms explored. 
    map_image = draw_map_png(cheat_mode and difficulty == "cheat")
    #Rooms Explored
    visited = [room.number for room in rooms.values() if room.explored]
    not_visited = [room.number for room in rooms.values() if not room.explored]
    rooms_visted = rooms_visted + "Rooms visited: " + str(visited) + "  Rooms to explore: " + str(not_visited) + "\n"
    #Inventory
    inventory1 = "" + playerOne.inventory_living()
    if playerOne.armors:
        inventory1 = inventory1 + "Wearing: " + ", ".join([f'{item.name}, that can protect from {item.protection} damage' for item in playerOne.armors]) + "\n"
         #Player status
    your_status = (f'Health: {int(playerOne.health)}  Wealth: {playerOne.wealth}  Points: {playerOne.points}  Kills: {playerOne.kills}')
    
    old_room = playerOne.current_room  

    image_gallery = (player_image or []) + (npc_image or []) + (room_image or []) + (monster_image or [] ) + (item_images or [])
    #print ("returning from talk_to_functions", player_image, npc_image, monster_image, room_image, item_images)

    return (image_gallery, map_image, 
        room_description.strip()+"\n"+room_riddles.strip(), # use room_description to show riddles
        room_contents.strip(), console.strip(), your_status, 
        rooms_visted.strip(), inventory1.strip(), 
        room_long_description, game_state_selector)

with gr.Blocks(theme=gr.themes.Default(font=[gr.themes.GoogleFont("IBM Plex Mono")])) as web_interface:
    if not list_of_saved_games:
        list_of_saved_games = ["No saved games"]
    if not list_of_llms:
        list_of_llms = ["No saved llms"]
    if not list_of_diffusers:
        list_of_diffusers = ["No saved diffusers"]

    with gr.Row():
        gr.Markdown("<h1 style='text-align: center;'>JMR's Colossal Cave Adventure</h1>")
    with gr.Row():
        with gr.Column():
            image_output = gr.Gallery(label="Room Image")
            text_input = gr.Textbox(autofocus=True, label="What do you want to do>")
            difficulty_selector = gr.Radio(["Cheat", "Easy", "Medium", "Hard"], value = "Easy", label="Difficulty")
            game_state_selector = gr.Radio(["Play-Game","Load-Game", "New-Game", "Save-Game"], label="Game State")
            
            with gr.Row():
                game_selector = gr.Dropdown(list_of_saved_games, visible=False, label="Select a saved game to load" , value = list_of_saved_games[0])
                name_of_game = gr.Textbox(visible = False, label="Name of game")  
                number_of_rooms = gr.Textbox(visible = False, label="Number of rooms")
                art_style = gr.Textbox(visible = False, label="Art Style")
            with gr.Row():
                llm_generation = gr.Checkbox(visible = False, value= True, label="Room & NPC Art")
                llm_monsters = gr.Checkbox(visible = False, value= True, label="Monster Art")
                llm_items = gr.Checkbox(visible = False, value= True, label="Weapon, Magic, Treasure Details")
            with gr.Row():
                llm_select = gr.Dropdown(list_of_llms, visible=False, label="Select LLM", value = list_of_llms[0])
                diff_select = gr.Dropdown(list_of_diffusers, visible=False, label="Select a Diffuser", value = list_of_diffusers[0])
                
            submit_button = gr.Button("Submit", visible=False)
        with gr.Column():
            room_desc_output = gr.Textbox(label="Room Description")
            contents_output = gr.Textbox(label="Room Contents")
            console_output = gr.Textbox(label="Console")
            status_output = gr.Textbox(label="Health, Wealth, Points, Killed")
            # Use a Row here to place map_output and map_image_output side by side
            with gr.Row():
                map_image_output = gr.Image(label=" ")
                with gr.Column():
                    rooms_explored = gr.Textbox(label="Rooms Explored")
                    inventory_out = gr.Textbox(label="Inventory")
            long_description_output = gr.Textbox(label="Room Long Description")
    # Conditionally show the game_selector
    def update_game_selector(game_state):
        if game_state == "Load-Game":
            return (gr.Dropdown(visible=True), gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False), 
        gr.Checkbox(visible=False), gr.Button(visible=True), gr.Dropdown(visible=True), gr.Dropdown(visible=True), gr.Textbox(visible=True))
        elif game_state == "Save-Game":
            return (gr.Dropdown(visible=False), gr.Textbox(visible=True), gr.Textbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False), 
        gr.Checkbox(visible=False), gr.Button(visible=True), gr.Dropdown(visible=False), gr.Dropdown(visible=False), gr.Textbox(visible=False))
        elif game_state == "New-Game":
            return (gr.Dropdown(visible=False), gr.Textbox(visible=True), gr.Textbox(visible=True), gr.Checkbox(visible=True), gr.Checkbox(visible=True), 
                gr.Checkbox(visible=True), gr.Button(visible=True), gr.Dropdown(visible=True), gr.Dropdown(visible=True), gr.Textbox(visible=True))
        elif game_state == "Play-Game":
            return (gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False), 
                gr.Checkbox(visible=False), gr.Button(visible=True), gr.Dropdown(visible=False), gr.Dropdown(visible=False), gr.Textbox(visible=False))
        else:
            return (gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Checkbox(visible=False) , gr.Checkbox(visible=False), 
        gr.Checkbox(visible=False), gr.Button(visible=False), gr.Dropdown(visible=False), gr.Dropdown(visible=False), gr.Textbox(visible=False))

    game_state_selector.change(update_game_selector, inputs=[game_state_selector], 
                               outputs=[game_selector, name_of_game, number_of_rooms, llm_generation, llm_monsters, llm_items, submit_button, llm_select, diff_select, art_style])
    
    # Now the function will be called when the submit button is clicked or Enter is pressed
    submit_inputs = [text_input, game_selector, difficulty_selector, game_state_selector, name_of_game, number_of_rooms, llm_generation, 
                     llm_monsters, llm_items, llm_select, diff_select, art_style]
    submit_outputs = [image_output, map_image_output, room_desc_output, contents_output, console_output, status_output, rooms_explored, 
                      inventory_out, long_description_output, game_state_selector]
    
    text_input.submit(  # Set up the submit event
        fn=talk_to_functions,
        inputs=submit_inputs,
        outputs=submit_outputs
    )
    submit_button.click(
        fn=talk_to_functions,
        inputs=submit_inputs,
        outputs=submit_outputs
    )

try:
    web_interface.launch(share=True)
except:
    web_interface.launch(share=False)   