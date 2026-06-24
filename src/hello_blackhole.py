import torch
import ttnn

def main():
    print("🚀 Démarrage du Hello World sur Tenstorrent Blackhole p150a...")

    # 1. Initialisation : Connexion à la carte
    # device_id=0 correspond à la première carte PCIe trouvée
    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    print(f"✅ Carte Blackhole (ID: {device_id}) initialisée avec succès.")

    # 2. Création des données sur le CPU (Hôte) via PyTorch
    # On crée une petite matrice de taille 32x32 avec des 1.0 (format bfloat16, très utilisé en IA)
    print("📦 Création d'un tenseur PyTorch sur le CPU...")
    torch_tensor = torch.ones((1, 1, 32, 32), dtype=torch.bfloat16)

    # 3. Transfert vers le Device (Carte Blackhole)
    # TILE_LAYOUT est le format mémoire natif optimisé des puces Tenstorrent (tuiles de 32x32)
    print("⬇️ Envoi du tenseur vers la carte Blackhole...")
    ttnn_tensor = ttnn.from_torch(
        torch_tensor, 
        dtype=ttnn.bfloat16, 
        layout=ttnn.TILE_LAYOUT, 
        device=device
    )

    # 4. Exécution de l'opération mathématique sur le matériel
    # Ici, on ajoute simplement le tenseur à lui-même (1.0 + 1.0 = 2.0)
    print("⚙️ Exécution de l'addition sur le hardware de la carte...")
    output_ttnn_tensor = ttnn.add(ttnn_tensor, ttnn_tensor)

    # 5. Rapatriement des données vers le CPU (Hôte)
    print("⬆️ Récupération du résultat depuis la carte...")
    output_torch_tensor = ttnn.to_torch(output_ttnn_tensor)

    # 6. Vérification du résultat
    # Si tout s'est bien passé, la première valeur devrait être 2.0
    valeur_test = output_torch_tensor[0, 0, 0, 0].item()
    print(f"📊 Résultat du calcul (attendu: 2.0) : {valeur_test}")
    
    if valeur_test == 2.0:
        print("🎉 Hello World réussi ! La carte Blackhole p150a répond parfaitement.")
    else:
        print("⚠️ Anomalie détectée dans le calcul.")

    # 7. Fermeture propre de la connexion matérielle
    ttnn.close_device(device)
    print("🔒 Connexion à la carte fermée.")

if __name__ == "__main__":
    main()