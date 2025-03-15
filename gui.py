import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox,filedialog
from v1_pipeline import v1_pipelin_youness

# Initialize the app
ctk.set_appearance_mode("dark")  # Options: "System", "Light", "Dark"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"
app = ctk.CTk()  

# --- STRUCTURE EN DEUX COLONNES ---
app.columnconfigure(0, weight=1)  # Colonne gauche (zone affichage)
app.columnconfigure(1, weight=2)  # Colonne droite (contenu principal)

fichiers_selectionnes = []
states=[]
nump_warp_g=1
radius_g=1

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
menu_outils.add_command(label="Paramètres")

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

# --- CADRE GAUCHE : Zone d'affichage des fichiers ---
frame_gauche = ctk.CTkFrame(app)
frame_gauche.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)

label_gauche = ctk.CTkLabel(frame_gauche, text="📂 Fichiers sélectionnés :", font=("Arial", 14, "bold"))
label_gauche.pack(pady=10)

text_zone = tk.Text(frame_gauche, height=20, wrap="word", font=("Arial", 12))
text_zone.pack(pady=5, padx=5, fill="both", expand=True)
text_zone.config(state="disabled")  # Désactive l'édition

app.title("Fully Customized Button")

# --- CADRE DROIT : Contenu principal (Bouton, Slider) ---
frame_droite = ctk.CTkFrame(app)
frame_droite.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)

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


# Fonction pour mettre à jour la valeur affichée
def update_value_radius(value):
    global radius
    value_label_radius.configure(text=f"radius: {int(float(value))}")
    radius=value

def update_value_nump(value):
    global nump_warp
    value_label_nump.configure(text=f"nump warp: {int(float(value))}")
    nump_warp=value

# Label pour afficher la valeur actuelle du slider
value_label_radius = ctk.CTkLabel(frame_droite, text="radius : 1", font=("Arial", 16))

value_label_nump = ctk.CTkLabel(frame_droite, text="nump warp : 1", font=("Arial", 16))


# Slider (curseur) de 1 à 30
radius = ctk.CTkSlider(
    frame_droite,
    from_=1,
    to=30,
    number_of_steps=29,
    command=update_value_radius
)
nump_warp = ctk.CTkSlider(
    frame_droite,
    from_=1,
    to=30,
    number_of_steps=29,
    command=update_value_nump
)
value_label_radius.pack(pady=10) # ajout du label
radius.set(1)  # Valeur initiale
radius.pack(pady=10)

value_label_nump.pack(pady=10)
nump_warp.set(1)  # Valeur initiale
nump_warp.pack(pady=10)

def execution():
    get_checkbox_states()
    v1_pipeline_youness(states,nump_warp=nump_warp_g,radius=radius_g)  # Appelle la fonction avec les valeurs correctes
 

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
    command=execution,
)
custom_button.pack(pady=20)

# Run the app
app.mainloop()
