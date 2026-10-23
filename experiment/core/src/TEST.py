import os
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext

import yaml
from ollama import Client
from openai import OpenAI

from multimodal_chatbot_ui import run_clara

class LLMmodel():
    """
    LLMmodel class creates the LLM model instance and handles interactions.
    """  
    """
    To correctly execute this code, you need to set the environment variable API_KEY with your API key.
    Execute the following commands in your terminal:
        echo "export OPENAI_API_KEY='insert_your_api_key_here'" >> ~/.bashrc
        source ~/.bashrc
    """
    def __init__(self, model, host, initial_prompt):
        """Constructor of the LLMmodel class.

        :param model: Model used for the LLM.
        :type model: str
        :param host: API host URL
        :type host: str
        :param initial_prompt: Prompt to initialize the LLM.
        :type initial_prompt: str
        """        
        self.model = model
        self.host = host
        self.initial_prompt = initial_prompt
        if model == "phi4:14b" or model == "Qwen3:30b":
            self.client = Client(host=self.host)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(base_url=self.host,
                                 api_key=api_key,)
    @staticmethod
    def LLM_worker(client:Client, model, command):
        """
        Extracts the response from the LLM based on the model used.

        :param client: LLM client instance.
        :type client: Client
        :param model: Model used for the LLM.
        :type model: str
        :param command: Command or messages to send to the LLM.
        :type command: list
        :return: Generated text from the LLM.
        :rtype: str
        """
        if model == "phi4:14b" or model == "Qwen3:30b":
            response = client.chat(model=model, messages=command, options={'temperature': 0.1})
            return response.message.content
        else:
            response = client.chat.completions.create(model=model, messages=command)
            return response.choices[0].message.content
    
    def send_to_LLM(self, conversation):
        """
        Sends the conversation and extra context to the LLM and returns the generated text.

        :param conversation: Conversation history.
        :type conversation: list
        :param extra_context: Additional context to provide to the LLM.
        :type extra_context: list
        :return: Generated text from the LLM.
        :rtype: str
        """
        messages = [self.initial_prompt]
        #messages.extend(extra_context)
        messages.extend(conversation)  
        generated_text = self.LLM_worker(self.client, self.model, messages)
        return generated_text


class ChatInterface():
    """
    ChatInterface class creates a GUI with the different models for user interaction.
    """
    def __init__(self, root, LLM, perceptions, needs, alignment_prompt, missions_prompt, drives_prompt, path):
        """
        Constructor of the ChatInterface class.

        :param root: Root window of the GUI.
        :type root: tk.Tk
        :param LLM: LLM model instance.
        :type LLM: LLMmodel
        :param perceptions: prompt with the perceptions to provide to the LLM.
        :type perceptions: str
        :param needs: prompt with the internal needs to provide to the LLM.
        :type needs: str
        :param alignment_prompt: prompt for the alignment LLM.
        :type alignment_prompt: str
        :param missions_prompt: prompt for the missions generation LLM.
        :type missions_prompt: str
        :param drives_prompt: prompt for the drives generation LLM.
        :type drives_prompt: str
        :param path: Path to save the conversation.
        :type path: str
        """
        self.root = root
        self.root.title("Chat Interface")
        self.LLM = LLM
        self.conversation = []
        self.final_purpose = None
        self.llm_mission = None
        self.LLM4drives = None
        self.drives = None
        self.awaiting_drive_feedback = False
        self.drive_conversation = []

        self.perceptions = perceptions
        self.needs = needs
        self.alignment_prompt = alignment_prompt
        self.missions_prompt = missions_prompt
        self.drives_prompt = drives_prompt
        self.path = path

        # Text area for conversation history
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=90, height=30, bg="#f4f4f4", font=("Arial", 11))
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Multiline input field
        self.user_input = tk.Text(root, width=70, height=4, font=("Arial", 12))
        self.user_input.grid(row=1, column=0, padx=10, pady=10)

        # Send button
        self.user_input.bind("<Control-Return>", self.send_message)
        self.send_button = tk.Button(root, text="Send", command=self.send_message, width=10, bg="#48C9F1", fg="white")
        self.send_button.grid(row=1, column=1, padx=10, pady=10)
        self.conversation = []
    
    def display_message(self, role, message):
        """
        Displays a message in the chat display area.

        :param role: Role of the message sender ('user' or 'assistant').
        :type role: str
        :param message: Message content to display.
        :type message: str

        """
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{role.capitalize()}: {message}\n\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.yview(tk.END)

    def send_message(self, event=None):
        """
        Recieves the user input, to send it to the alignment LLM, and displays the response. Calls the generate missions and drives methods.

        :param event: Event that triggered the method, defaults to None.
        :type event: tk.Event or NoneType

        """
        user_msg = self.user_input.get("1.0", tk.END).strip()
        if not user_msg:
            return

        self.conversation.append({'role': 'user', 'content': user_msg})
        self.display_message('user', user_msg)
        self.user_input.delete("1.0", tk.END)

        try:
            objects = [{"role": "system", "content": str(self.perceptions["objects"])}]
            alignment_conversation = self.conversation + objects
            reply = self.LLM.send_to_LLM(alignment_conversation)
            self.conversation.append({'role': 'assistant', 'content': reply})
            self.display_message('assistant', reply)
            save_conversation(self.conversation, self.path)

            if any(marker in reply for marker in (
                    "Final description", "Final message", "Final answer")):
                self.final_purpose = reply
                messagebox.showinfo("Info", "Final description received. Generating missions...")
                self.generate_missions(objects)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def generate_missions(self, objects):
        """
        Sends the final purpose and internal needs to the LLM model and generates the missions.

        :param objects: Description of objects to provide to the LLM.
        :type objects: str

        """
        needs_list = [{"role": "system", "content": str(self.needs["needs"])}]
        purpose = [{"role": "system", "content": str(self.final_purpose)}]
        combined_data = purpose + objects + needs_list
        model, host, missions_p = load_configuration(self.missions_prompt)
        LLM4missions = LLMmodel(model, host, missions_p)

        try:
            mission_response = LLM4missions.send_to_LLM(combined_data)
            self.llm_mission = mission_response
            self.conversation.append({'role': 'assistant', 'content': mission_response})
            self.display_message('assistant', mission_response)
            save_conversation(self.conversation, self.path)

            messagebox.showinfo("Info", "Missions generated. Generating drives...")
            self.generate_drives(objects)
        except Exception as e:
            messagebox.showerror("Mission Generation Error", str(e))

    def generate_drives(self, objects):
        """
        Sends the final purpose and the list of missions to the LLM model and generates the drives.

        :param objects: Description of objects to provide to the LLM.
        :type objects: str

        """
        model, host, drives_p = load_configuration(self.drives_prompt)
        LLM4drives = LLMmodel(model, host, drives_p)
        self.LLM4drives = LLM4drives
        first_prompt = self.final_purpose + "\n\n" + self.llm_mission
        self.drive_conversation.extend([{'role': 'user', 'content': first_prompt}] + objects)
        try:
            drive_response = LLM4drives.send_to_LLM(self.drive_conversation)
            self.conversation.append({'role': 'assistant', 'content': drive_response})
            self.drive_conversation.append({'role': 'assistant', 'content': drive_response})
            self.display_message('assistant', drive_response)
            save_conversation(self.conversation, self.path)

            if "Final drives" in drive_response:
                    self.drives = drive_response
                    messagebox.showinfo("Drives", "Final drives received. Interaction complete.")
                    self.root.after(500, self.root.destroy)
            else:
                self.awaiting_drive_feedback = True
                self.send_button.configure(command=lambda: self.send_drive_feedback(objects))
                self.user_input.bind('<Control-Return>', lambda event: self.send_drive_feedback(objects))
                
        except Exception as e:
            messagebox.showerror("Drives Generation Error", str(e))

    def send_drive_feedback(self, objects):
        """
        Recieves the feedback from the user to refine the drives if needed and finishes the interaction when the final drives are accepted.

        :param objects: Description of objects to provide to the LLM.
        :type objects: str

        """
        if not self.awaiting_drive_feedback:
            return
        feedback = self.user_input.get("1.0", tk.END).strip()
        if not feedback:
            return

        self.user_input.delete("1.0", tk.END)
        self.display_message('user', feedback)
        self.drive_conversation.append({'role': 'user', 'content': feedback})
        self.drive_conversation.extend(objects)
        self.conversation.append({'role': 'user', 'content': feedback})

        try:
            drive_response = self.LLM4drives.send_to_LLM(self.drive_conversation)
            self.drive_conversation.append({'role': 'assistant', 'content': drive_response})
            self.conversation.append({'role': 'assistant', 'content': drive_response})
            self.display_message('assistant', drive_response)
            save_conversation(self.conversation, self.path)

            if "Final answer" in drive_response:
                self.drives = drive_response
                self.awaiting_drive_feedback = False
                messagebox.showinfo("Drives", "Final drives received. Interaction complete.")
                self.root.after(10000, self.root.destroy)
            else:
                # Stay in feedback loop
                self.awaiting_drive_feedback = True
                self.send_button.configure(command=lambda: self.send_drive_feedback(objects))
                self.user_input.bind('<Control-Return>', lambda event: self.send_drive_feedback(objects))

        except Exception as e:
            messagebox.showerror("Feedback Error", str(e))


class MissionsDrivesforCLARAChatInterface():
    """Chat flow that performs alignment and mission generation only."""

    def __init__(self, root, perceptions, needs, missions, drives, path):
            """
            Constructor of the ChatInterface class.
    
            :param root: Root window of the GUI.
            :type root: tk.Tk
            :param LLM: LLM model instance.
            :type LLM: LLMmodel
            :param perceptions: prompt with the perceptions to provide to the LLM.
            :type perceptions: str
            :param needs: prompt with the internal needs to provide to the LLM.
            :type needs: str
            :param alignment_prompt: prompt for the alignment LLM.
            :type alignment_prompt: str
            :param weights_prompt: prompt for the weights generation LLM.
            :type weights_prompt: str
            :param path: Path to save the conversation.
            :type path: str
            """
            
            self.conversation = []
            self.final_purpose = None
            self.llm_mission = None
            self.LLM4drives = None
            self.drives = None

            self.perceptions = perceptions
            self.needs = needs
            self.missions_prompt = missions
            self.drives_prompt = drives
            self.path = path

    def send_message(self, event=None):
            """
            Recieves the user input, to send it to the alignment LLM, and displays the response. Calls the generate missions and drives methods.
    
            :param event: Event that triggered the method, defaults to None.
            :type event: tk.Event or NoneType
    
            """
            objects = [{"role": "system", "content": str(self.perceptions["objects"])}]
            self.generate_missions(objects)
           
    def generate_missions(self, objects):
        """
        Sends the final purpose and internal needs to the LLM model and generates the missions.

        :param objects: Description of objects to provide to the LLM.
        :type objects: str

        """
        needs_list = [{"role": "system", "content": str(self.needs["needs"])}]
        purpose = [{"role": "system", "content": str(self.final_purpose)}]
        combined_data = purpose + objects + needs_list
        model, host, missions_p = load_configuration(self.missions_prompt)
        LLM4missions = LLMmodel(model, host, missions_p)

        try:
            mission_response = LLM4missions.send_to_LLM(combined_data)
            self.llm_mission = mission_response
            self.conversation.append({'role': 'assistant', 'content': mission_response})
            save_conversation(self.conversation, self.path)
            self.generate_drives(objects)
        except Exception as e:
            print("Mission Generation Error")

    def generate_drives(self, objects):
        """
        Sends the final purpose and the list of missions to the LLM model and generates the drives.

        :param objects: Description of objects to provide to the LLM.
        :type objects: str

        """
        model, host, drives_p = load_configuration(self.drives_prompt)
        LLM4drives = LLMmodel(model, host, drives_p)
        self.LLM4drives = LLM4drives
        first_prompt = self.final_purpose + "\n\n" + self.llm_mission
        self.drive_conversation.extend([{'role': 'user', 'content': first_prompt}] + objects)
        try:
            drive_response = LLM4drives.send_to_LLM(self.drive_conversation)
            self.conversation.append({'role': 'assistant', 'content': drive_response})
            self.drive_conversation.append({'role': 'assistant', 'content': drive_response})
            self.display_message('assistant', drive_response)
            save_conversation(self.conversation, self.path)
            self.drives = drive_response
            #print(self.drives)
                
        except Exception as e:
            print("Drives Generation Error")

def init_conversation_file():
    """
    It finds the first available filename and saves it in a global variable.

    :return: Path to the conversation file.
    :rtype: str
    """
    base_name = "conversation_"
    extension = ".yaml"
    dir = os.getcwd()
    i = 1
    while True:
        file_name = f"{base_name}{i}{extension}"
        file_path = os.path.join(dir, file_name)
        if not os.path.exists(file_path):
            return file_path
        i += 1

def save_conversation(conversation, file_path):
    """
    Writes to the same file chosen at init.

    :param conversation: Conversation to save.
    :type conversation: list
    :param file_path: Path to the conversation file.
    :type file_path: str

    """
    if file_path is None:
        raise RuntimeError("You must call init_conversation_file() before save_conversation()")

    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(conversation, file, allow_unicode=True)

def load_configuration(config_prompt):
    """
    Extracts the model, host, and initial prompt from the configuration dictionary.

    :param config_prompt: Configuration dictionary.
    :type config_prompt: dict
    :return: model, host, initial_prompt
    :rtype: tuple[str, str, str]
    """
    model = config_prompt["model"]
    host = config_prompt["host"]
    initial_prompt = config_prompt["initial_prompt"]
    return model, host, initial_prompt

def _load_prompts(prompts_file, process):
        """
        Extract the different prompts from the YAML file.

        :return: alignment_prompt, drives_prompt, missions_prompt, needs, perceptions
        :rtype: tuple[str, str, str, str, str]
        """
        with open(prompts_file, "r", encoding="utf-8") as file:
            prompts = yaml.load(file, Loader=yaml.FullLoader)
            if process == "full":
                alignment_prompt = prompts.get('alignment_prompt')
                drives_prompt = prompts.get('drives_prompt')
                missions_prompt = prompts.get('missions_prompt')
                needs = prompts.get('internal_needs')
                perceptions = prompts.get('perceptions')
                return alignment_prompt, drives_prompt, missions_prompt, needs, perceptions
            elif process == "joined_paper":
                drives_prompt = prompts.get('drives_prompt')
                missions_prompt = prompts.get('missions_prompt')
                needs = prompts.get('internal_needs')
                perceptions = prompts.get('perceptions')
                return drives_prompt, missions_prompt, needs, perceptions



def extract_missions_and_drives(text):
    """
    Extracts missions and drives from the LLM response text.

    :param text: LLM response text.
    :type text: str or list
    :return: Extracted missions and drives.
    :rtype: tuple[list, list]M = Non
    """    
    if text is None:
        return [], []
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)
        
    missions = []
    drives = []
    mission_blocks = re.findall(
        r'(Mission\d+:\s*\[.*?\]\s*Drive:\s*.*?)(?=(?:\n\s*\n|$))',
        text, re.DOTALL
    )

    for block in mission_blocks:
        tag_match = re.search(r'Mission\d+:\s*\[([^\],]+),\s*([0-9.]+)\]', block)
        drive_match = re.search(r'Drive:\s*(.*)', block)
        if tag_match and drive_match:
            mission_tag = tag_match.group(1)
            mission_value = float(tag_match.group(2))
            drive = drive_match.group(1).strip()

            missions.append([mission_tag, mission_value])
            drives.append(drive)

    return missions, drives
def extract_missions_weights(text):
    """
    Extracts missions and weights from the LLM response text.

    :param text: LLM response text.
    :type text: str or list
    :return: Extracted missions and weights.
    :rtype: list
    """    
    if text is None:
        return []
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)
    missions_weights = []
    mission_blocks = re.findall(
        r"\[\s*['\"]?([A-Za-z0-9_]+)['\"]?\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]",
        text, re.DOTALL
    )

    for block in mission_blocks:
        mission_tag = block[0]
        weight_value = float(block[1].strip())
        missions_weights.append([mission_tag, weight_value])

    return missions_weights

def interface(process):
        """
        Starts the interface that interacts with the user to align the human's purpose with the robot's purpose.

        :return: final_answer, missions, drives
        :rtype: tuple[str, list, list]
        """
        path = init_conversation_file()
        file_path = Path(__file__).parent / "Prompts.yaml"
        if process == "full":
            alignment, drives, missions, needs, perceptions = _load_prompts(file_path, process)
            model, host, initial_prompt = load_configuration(alignment)
            LLM = LLMmodel(model, host, initial_prompt)
            root = tk.Tk()
            results = ChatInterface(root, LLM, perceptions, needs, alignment, missions, drives, path)
            print(results)
            root.mainloop()
        elif process == "joined_paper":
               
            drives, missions, needs, perceptions = _load_prompts(file_path, process)
            # CLARA: conversation + image
            session = run_clara()
            if not session.image_path:
                print("CLARA didn't generate an image.")
                return None

            results = MissionsDrivesforCLARAChatInterface(None, perceptions, needs, missions, drives, path)
            results.final_purpose = session.final_purpose
            results.image_path = session.image_path
            results.drive_conversation = []                        
            results.display_message = lambda role, message: None  
            results.send_message()                                 
        else:
            raise ValueError("process must be 'full' or 'joined_paper'")
        return results

if __name__ == "__main__":
    process = "joined_paper"  # Change to "full" for the full process
    final_answer = interface(process=process)
    if final_answer is not None:
        print(final_answer.drives)



    #print("MISSIONS:\n", missions)
    #print("\nDRIVES:\n", drives)
