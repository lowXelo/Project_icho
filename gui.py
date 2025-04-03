import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox,filedialog
from v1_pipeline import v1_pipelin_youness,v1_direct
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import numpy as np


# Initialize the app
ctk.set_appearance_mode("dark")  # Options: "System", "Light", "Dark"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"
app = ctk.CTk()  
app.title("HIA")

# --- STRUCTURE EN DEUX COLONNES ---
app.columnconfigure(0, weight=1)  # Colonne gauche (zone affichage)
app.columnconfigure(1, weight=2)  # Colonne droite (contenu principal)

fichiers_selectionnes = []
states=[]
pipline_result={}

# --- FONCTIONS ---
def ouvrir_fichier():
    global fichiers_selectionnes
    fichiers_selectionnes = filedialog.askopenfilenames(
        title="Sélectionnez plusieurs fichiers",
        filetypes=[("Tous les fichiers", "*.*"),
                   ("Fichiers texte", "*.txt"),
                   ("Images", "*.png;*.jpg;*.jpeg")]
    )

    if fichiers_selectionnes:  # Si des fichiers sont sélectionnés
        messagebox.showinfo("Fichiers sélectionnés", f"{len(fichiers_selectionnes)} fichier(s) sélectionné(s) !")
        afficher_chemins()

# --- AFFICHER LES CHEMINS SELECTIONNÉS ---
def afficher_chemins():
    text_zone.config(state="normal")  # Débloque l'édition
    text_zone.delete("1.0", tk.END)  # Efface l'ancien contenu
    for fichier in fichiers_selectionnes:
        text_zone.insert(tk.END, f"{fichier}\n")  # Ajoute chaque fichier
    text_zone.config(state="disabled")  # Bloque l'édition


def enregistrer_fichier():
    messagebox.showinfo("Enregistrer", "Fichier enregistré !")

def quitter_application():
    app.quit()

def a_propos():
    messagebox.showinfo("À propos", "Cette application utilise CustomTkinter")

# --- BARRE DE MENU ---
menu_bar = tk.Menu(app)

# --- MENU FICHIER ---
menu_fichier = tk.Menu(menu_bar, tearoff=0)
menu_fichier.add_command(label="Ouvrir", command=ouvrir_fichier)
menu_fichier.add_command(label="Enregistrer", command=enregistrer_fichier)
menu_fichier.add_separator()
menu_fichier.add_command(label="Quitter", command=quitter_application)

# --- MENU ÉDITION ---
menu_edition = tk.Menu(menu_bar, tearoff=0)
menu_edition.add_command(label="Couper")
menu_edition.add_command(label="Copier")
menu_edition.add_command(label="Coller")

# --- MENU OUTILS ---
menu_outils = tk.Menu(menu_bar, tearoff=0)
functions_dict = {
    "V1 search": lambda: update_interface("search"),
    "V1 calculate": lambda: update_interface("use"),
}

def update_interface(mode):
    """Updates the UI based on the selected function from the 'Outils' menu."""
    for widget in frame_droite.winfo_children():
        widget.destroy()  # Clear current UI elements in the right frame

    label_droite = ctk.CTkLabel(frame_droite, text=f"Mode: {mode}", font=("Arial", 14, "bold"))
    label_droite.pack(pady=10)

    if mode == "search":
# --- CADRE DROIT : Contenu principal (Bouton, Slider) ---
        label_droite = ctk.CTkLabel(frame_droite, text="🎛️ Actions :", font=("Arial", 14, "bold"))
        label_droite.pack(pady=10)

        # --- CHECKBOXES ---
        checkboxes = []
        checkbox_vars = []
        checkboxes_names = ["box plot","moyenne","variance","temps de traitement"]
        label_checkbox = ctk.CTkLabel(frame_droite, text="✅ Sélectionnez des options :", font=("Arial", 14))
        label_checkbox.pack(pady=10)

        # Création de 4 checkboxes
        for i in range(4):
            var = ctk.IntVar(value=0)  # 0 par défaut
            checkbox = ctk.CTkCheckBox(frame_droite, text=checkboxes_names[i], variable=var)
            checkbox.pack(pady=2, anchor="w")
            checkboxes.append(checkbox)
            checkbox_vars.append(var)

        # --- FONCTION POUR RÉCUPÉRER L'ÉTAT DES CHECKBOXES ---
        def get_checkbox_states():
            global states 
            states = [var.get() for var in checkbox_vars]
            messagebox.showinfo("État des Checkboxes", f"État en binaire : {states}")

        nump_warp_g=2
        radius_g=2

        # Fonction pour mettre à jour la valeur affichée
        def update_value_radius(value):
            global radius_g
            value_label_radius.configure(text=f"radius: {int(float(value))}")
            radius_g=value+1

        def update_value_nump(value):
            global nump_warp_g
            value_label_nump.configure(text=f"nump warp: {int(float(value))}")
            nump_warp_g=value+1

        # Label pour afficher la valeur actuelle du slider
        value_label_radius = ctk.CTkLabel(frame_droite, text="radius : 1", font=("Arial", 16))

        value_label_nump = ctk.CTkLabel(frame_droite, text="nump warp : 1", font=("Arial", 16))


        # Slider (curseur) de 1 à 30
        slider_radius = ctk.CTkSlider(
            frame_droite,
            from_=1,
            to=30,
            number_of_steps=29,
            command=update_value_radius
        )
        slider_nump_warp = ctk.CTkSlider(
            frame_droite,
            from_=1,
            to=30,
            number_of_steps=29,
            command=update_value_nump
        )
        value_label_radius.pack(pady=10) # ajout du label
        slider_radius.set(1)  # Valeur initiale
        slider_radius.pack(pady=10)

        value_label_nump.pack(pady=10)
        slider_nump_warp.set(1)  # Valeur initiale
        slider_nump_warp.pack(pady=10)

        # Create a progress bar
        progress_bar = ctk.CTkProgressBar(app)
        progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10) 

        # Set the initial progress to 0
        progress_bar.set(0)

        def plot_results(returned_data):
            for file_key, matrices in returned_data.items():
                valid_matrices = {k: v for k, v in matrices.items() if k != "Best combination"}
                num_matrices = len(valid_matrices)

                rows, cols = 2, 2  # Fixed 2x2 grid
                fig, axes = plt.subplots(rows, cols, figsize=(10, 10))  
                axes = axes.flatten()  # Convert to a list for easier iteration

                for idx, (matrix_name, matrix) in enumerate(valid_matrices.items()):
                    ax = axes[idx]
                    
                    if matrix_name == "data":
                        ax.boxplot(matrix)
                        ax.set_xlabel("Columns")
                        ax.set_ylabel("Rows")
                    else:
                        im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
                        fig.colorbar(im, ax=ax)
                        ax.set_xlabel("Columns")
                        ax.set_ylabel("Rows")
                    
                    ax.set_title(matrix_name)
                
                # Hide any unused subplots
                for j in range(num_matrices, len(axes)):
                    fig.delaxes(axes[j])

                fig.suptitle(f"File: {file_key}", fontsize=14, fontweight="bold")
                plt.tight_layout()
                plt.show(block=False)  # Display the plot
                    
        # Function to start processing in a separate process
        def start_processing():
            result_queue={}
            global process
            global nump_warp_g,radius_g
            queue = Queue()
            result_queue = Queue()
            get_checkbox_states()

            process = Process(target=v1_pipelin_youness, args=(states, queue,result_queue,fichiers_selectionnes,int(nump_warp_g),int(radius_g)))
            process.start()

            # Update progress bar dynamically
            def update_progress():
                if not queue.empty():
                    progress_value = queue.get()
                    progress_bar.set(progress_value/100)  # Correct usage in CustomTkinter
                    if progress_value==100:
                        pipline_result=result_queue.get()
                        print(pipline_result)
                        print(nump_warp_g,radius_g)
                        plot_results(pipline_result)

                if process.is_alive():
                    app.after(100, update_progress)  # Continue updating       

            update_progress()
        
            

            # Bouton personnalisé
        custom_button = ctk.CTkButton(
            frame_droite,
            text="Lancer le traitement",
            width=200,
            height=50,
            fg_color="dodgerblue",
            hover_color="deepskyblue",
            text_color="white",
            corner_radius=10,
            border_width=2,
            border_color="white",
            font=("Arial", 16, "bold"),
            command=start_processing,
            )
        custom_button.pack(pady=20)
            
    elif mode == "use":
        # --- CHECKBOXES ---
        checkboxes = []
        checkbox_vars = []
        checkboxes_names = ["old_cc","new_cc","image before","image after"]
        label_checkbox = ctk.CTkLabel(frame_droite, text="✅ Sélectionnez des options :", font=("Arial", 14))
        label_checkbox.pack(pady=10)

        # Création de 4 checkboxes
        for i in range(4):
            var = ctk.IntVar(value=0)  # 0 par défaut
            checkbox = ctk.CTkCheckBox(frame_droite, text=checkboxes_names[i], variable=var)
            checkbox.pack(pady=2, anchor="w")
            checkboxes.append(checkbox)
            checkbox_vars.append(var)

        # --- FONCTION POUR RÉCUPÉRER L'ÉTAT DES CHECKBOXES ---
        def get_checkbox_states():
            global states 
            states = [var.get() for var in checkbox_vars]
            messagebox.showinfo("État des Checkboxes", f"État en binaire : {states}")

        radius_entry = ctk.CTkLabel(frame_droite, text="Radius :")
        radius_entry.pack(pady=5)

        radius_entry = ctk.CTkEntry(frame_droite)
        radius_entry.pack(pady=5)

        Nump_warp_entry= ctk.CTkLabel(frame_droite, text="Nump warp:")
        Nump_warp_entry.pack(pady=5)

        Nump_warp_entry = ctk.CTkEntry(frame_droite)
        Nump_warp_entry.pack(pady=5)

        nump_warp_d=2
        radius_d=2
        # Button to Get Values
        def get_values():
            global nump_warp_d,radius_d
            nump_warp_d=Nump_warp_entry.get()
            radius_d=radius_entry.get()

        def plot_results_direct(returned_data):
            for file_key, matrices in returned_data.items():
                valid_matrices = {k: v for k, v in matrices.items() if k != "Best combination"}
                num_matrices = len(valid_matrices)

                rows, cols = 2, 2  # Fixed 2x2 grid
                fig, axes = plt.subplots(rows, cols, figsize=(10, 10))  
                axes = axes.flatten()  # Convert to a list for easier iteration

                for idx, (matrix_name, matrix) in enumerate(valid_matrices.items()):
                    ax = axes[idx]
                    
                    if matrix_name == "data_old":
                        ax.boxplot(matrix)

                    elif matrix_name == "data_new":
                        ax.boxplot(matrix)
                    
                    elif matrix_name == "image_before":
                        ax.imshow(matrix)
                        print("one")
                        print(len(matrix))
                    else:
                        ax.imshow(np.array(matrix))
                        print("twp")
                        print(len(matrix[0]))
                        

                    
                    # ax.set_title(matrix_name)
            for j in range(num_matrices, len(axes)):
                fig.delaxes(axes[j])

            fig.suptitle(f"File: {file_key}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show(block=False)  # Display the plot

        def start_processing_direct():
            result_queue={}
            global process
            queue = Queue()
            result_queue = Queue()
            get_checkbox_states()
            get_values()
            process = Process(target=v1_direct, args=(states, queue,result_queue,fichiers_selectionnes,int(nump_warp_d),int(radius_d)))
            process.start()

            # Update progress bar dynamically
            def update_progress():
                if not queue.empty():
                    progress_value = queue.get()
                    progress_bar.set(progress_value/100)  # Correct usage in CustomTkinter
                    if progress_value==100:
                        pipline_result=result_queue.get()
                        plot_results_direct(pipline_result)

                if process.is_alive():
                    app.after(100, update_progress)  # Continue updating       

            update_progress()        

            # Bouton personnalisé
        custom_button = ctk.CTkButton(
            frame_droite,
            text="Lancer le traitement",
            width=200,
            height=50,
            fg_color="dodgerblue",
            hover_color="deepskyblue",
            text_color="white",
            corner_radius=10,
            border_width=2,
            border_color="white",
            font=("Arial", 16, "bold"),
            command=start_processing_direct,
            )   

                # Create a progress bar
        progress_bar = ctk.CTkProgressBar(app)
        progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10) 

        # Set the initial progress to 0
        progress_bar.set(0)

        plot_button = ctk.CTkButton(frame_droite, text="Afficher Graphiques", command=start_processing_direct)
        plot_button.pack(pady=10)

for name, func in functions_dict.items():
    menu_outils.add_command(label=name, command=func)

# --- MENU AIDE ---
menu_aide = tk.Menu(menu_bar, tearoff=0)
menu_aide.add_command(label="À propos", command=a_propos)

# Ajout des menus à la barre
menu_bar.add_cascade(label="Fichier", menu=menu_fichier)
menu_bar.add_cascade(label="Édition", menu=menu_edition)
menu_bar.add_cascade(label="Outils", menu=menu_outils)
menu_bar.add_cascade(label="Aide", menu=menu_aide)

# Assigner la barre de menu à la fenêtre principale
app.config(menu=menu_bar)

# --- CADRE DROITE : Zone d'affichage des fichiers ---
frame_droite = ctk.CTkFrame(app)
frame_droite.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)

# --- CADRE GAUCHE : Zone d'affichage des fichiers ---
frame_gauche = ctk.CTkFrame(app)
frame_gauche.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)

label_gauche = ctk.CTkLabel(frame_gauche, text="📂 Fichiers sélectionnés :", font=("Arial", 14, "bold"))
label_gauche.pack(pady=10)

text_zone = tk.Text(frame_gauche, height=20, wrap="word", font=("Arial", 12))
text_zone.pack(pady=5, padx=5, fill="both", expand=True)
text_zone.config(state="disabled")  # Désactive l'édition


if __name__ == "__main__":
    # Run the app
    app.mainloop()
