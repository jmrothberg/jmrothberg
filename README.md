
This project is an interactive text-based adventure game, reminiscent of classic games like "Colossal Cave Adventure", enhanced with the power of Language Large Models (LLM) for natural language processing and diffusion models for image generation, allowing for a rich narrative and visual experience. The game dynamically generates descriptions, dialogues, riddles, and artwork during gameplay.

## Setup

To run the game, you need to have Python installed on your system. Additionally, you will need to install several packages and APIs, and have access to pre-trained LLMs and diffusion models. The game is designed for Unix-like systems with a GPU capable of running PyTorch and associated deep learning models; however, it also includes support for macOS devices with MPS (Metal Performance Shaders).

### Dependencies

Install the required Python libraries from the command line with `pip`:

```sh
pip install torch torchvision torchaudio gradio sentence-transformers fuzzywuzzy python-Levenshtein diffusers
```

### Language Large Models (LLM)

This game utilizes pre-trained language models (Llama or your choice) which should be placed in a specified directory. The game will ask you to choose the path where the LLMs are stored if it can't find them at the provided location.

### Diffusion Models

Ongoing advances in image diffusion models allow the game to generate visual content dynamically. Pre-trained models for text-to-image translation must be stored in a specific directory, which the game will request if not found.

### Models and Data

Ensure that your model files and game data are accessible by placing them in a predefined location or selecting the appropriate path when prompted by the game.

### Running the Game

With all dependencies installed and models placed in the correct directories, navigate to the game's directory in the terminal and execute the following command to start the session:

```sh
python colossal_adventure.py
```

## Gameplay

The game follows basic command-line interactions:

- `go <direction>`: Move through the game's world. Directions include north, south, east, and west.
- `take <item>`: Pick up an item from the current room.
- `leave <item>`: Drop an item in the current room.
- `use <item>`: Utilize an item in your inventory, such as magic items for healing or solving puzzles.
- `study <item>`: Examine an item in the room closer.
- `attack <monster> with <weapon>`: Engage in combat with a monster using a weapon.
- `trade <item> for <item>`: Exchange items with an NPC.
- `talk <text>`: Start a dialogue with an NPC.
- `details`: Request the LLM to provide an updated description of the room.
- `draw`: Use the diffusion model to generate and update the current room's image.
- `solve <item1> and <item2>`: Use magic items from your inventory to solve a room's riddle.
- `magicword <room_number>`: If in cheat mode, opens up magical passages in a room.
- `health <value>`: If in cheat mode, sets your health to a specified value.
- `help`: Display the list of valid commands.

Note: The magic word for enabling cheat mode is generated uniquely for each game and will be revealed through gameplay.

### Interactive UI (optional)

For a friendly user interface, start a Gradio session that provides text boxes and buttons for easier interaction with the game.

## Conclusion

This custom adventure game demonstrates the fusion of classic gameplay with cutting-edge AI technologies, expanding the possibilities of what such games can offer. Players can experience an ever-changing adventure with unique surprises each time they play. Use this guide to set up and explore the enchanted world of this advanced Colossal Cave Adventure.


- 👋 Hi, I’m @jmrothberg
- 👀 I’m interested in molecular biology, healthcare, and the environment
- 🌱 I’m currently learning ... python, LLMs, Nueral networks.
- 💞️ I’m looking to collaborate on ... A Game of life with Genetics to control cell fate, and neurons that self wire into working networks that can solve MNEST, and other problems.
- 📫 How to reach me ... jonathan.rothberg@gmail.com

<!---
jmrothberg/jmrothberg is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
