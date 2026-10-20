BENCHMARK_PROFILES = {
    # --- 1. SQUEEZENET (~1.2M params - TinyML) ---
    ("squeezenet", "mnist"): {
        "resolution": (32, 32),
        "unfreeze_strategy": "full",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,             
        "epochs": 1,
        "max_train_samples": 1000,    
        "max_test_samples": 500,
        "description": "TinyML I: Clasificación ultra-rápida de dígitos."
    },
    ("squeezenet", "svhn"): {
        "resolution": (64, 64),
        "unfreeze_strategy": "full",
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 1,
        "max_train_samples": 1200,    
        "max_test_samples": 500,
        "description": "TinyML II: Detección de caracteres urbanos y matrículas."
    },
    ("squeezenet", "cifar10"): {
        "resolution": (64, 64),
        "unfreeze_strategy": "full",
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 1,
        "max_train_samples": 1500,   
        "max_test_samples": 500,
        "description": "TinyML III: Objetos genéricos en micro-rovers."
    },

    # --- 2. MOBILENET-V3-SMALL (~2.5M params - Ligero) ---
    ("mobilenet_v3_small", "mnist"): {
        "resolution": (32, 32),
        "unfreeze_strategy": "full",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "epochs": 1,
        "max_train_samples": 1000,
        "max_test_samples": 500,
        "description": "Ligero I: Inferencia de bajo consumo en señales numéricas."
    },
    ("mobilenet_v3_small", "svhn"): {
        "resolution": (64, 64),
        "unfreeze_strategy": "full",
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 1,
        "max_train_samples": 1200,
        "max_test_samples": 500,
        "description": "Ligero II: Percepción urbana ligera."
    },
    ("mobilenet_v3_small", "eurosat"): {
        "resolution": (64, 64),
        "unfreeze_strategy": "full",
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 1,
        "max_train_samples": 1600,   
        "max_test_samples": 400,
        "description": "Ligero III: Teledetección y cobertura terrestre en drones."
    },

    # --- 3. MOBILENET-V2 (~3.5M params - Medio I) ---
    ("mobilenet_v2", "svhn"): {
        "resolution": (64, 64),
        "unfreeze_strategy": "full",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 1,
        "max_train_samples": 1200,
        "max_test_samples": 500,
        "description": "Medio I: Reconocimiento continuo de hitos viales."
    },
    ("mobilenet_v2", "gtsrb"): {
        "resolution": (64, 64),
        "unfreeze_strategy": "full",
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 1,
        "max_train_samples": 2150,    
        "max_test_samples": 860,
        "description": "Medio II: ADAS - Señales de tráfico en tiempo real."
    },
    ("mobilenet_v2", "plantvillage"): {
        "resolution": (128, 128),
        "unfreeze_strategy": "full",
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 1,
        "max_train_samples": 1900,  
        "max_test_samples": 760,
        "description": "Medio III: Diagnóstico foliar y patología agrícola."
    },

    # --- 4. EFFICIENTNET-B0 (~5.3M params - Medio II) ---
    ("efficientnet", "gtsrb"): {
        "resolution": (64, 64),
        "unfreeze_strategy": "full",
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 1,
        "max_train_samples": 2150,
        "max_test_samples": 860,
        "description": "Medio IV: Detección multiclase de alta precisión vial."
    },
    ("efficientnet", "pets"): {
        "resolution": (128, 128),
        "unfreeze_strategy": "full",
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "batch_size": 16,            
        "epochs": 1,
        "max_train_samples": 1480,
        "max_test_samples": 370,
        "description": "Medio V: Monitoreo ganadero y fenotipado animal."
    },
    ("efficientnet", "plantvillage"): {
        "resolution": (128, 128),
        "unfreeze_strategy": "full",
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 1,
        "max_train_samples": 1900,
        "max_test_samples": 760,
        "description": "Medio VI: Clasificación de 38 tipos de cultivos."
    },

    # --- 5. RESNET-18 (~11.7M params - Pesado I) ---
    ("resnet18", "pets"): {
        "resolution": (128, 128),
        "unfreeze_strategy": "partial",
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "epochs": 1,
        "max_train_samples": 1480,
        "max_test_samples": 370,
        "description": "Pesado I: Representación profunda de grano fino en fauna."
    },
    ("resnet18", "cifar100"): {
        "resolution": (128, 128),
        "unfreeze_strategy": "partial",
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 1,
        "max_train_samples": 2500,    
        "max_test_samples": 1000,
        "description": "Pesado II: Inventario logístico de almacén (100 SKUs)."
    },

    # --- 6. RESNET-34 (~21.3M params - Pesado II) ---
    ("resnet34", "cifar100"): {
        "resolution": (128, 128),
        "unfreeze_strategy": "head_only",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 1,
        "max_train_samples": 2500,
        "max_test_samples": 1000,
        "description": "Pesado III: Linear probing masivo sobre 100 categorías."
    },
    ("resnet34", "flowers102"): {
        "resolution": (160, 160),    
        "unfreeze_strategy": "head_only",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 16,             
        "epochs": 1,
        "max_train_samples": 2040,    
        "max_test_samples": 510,
        "description": "Pesado IV: Discriminación botánica fina a 160x160."
    }
}
