# tests/tt_device_open.py
import ttnn

print("Ouverture du device Blackhole...")
device = ttnn.open_device(device_id=0)

# Attendu dans les logs :
# Device | INFO | Opening user mode device driver
# Metal  | INFO | Initializing device 0
# Metal  | INFO | AI CLK for device 0 is: 1000 MHz

print(f"Device ouvert : {device}")
ttnn.close_device(device)
print("Device fermé ✅")