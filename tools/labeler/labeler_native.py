import sys
import os
import json

# Safe import for PyVista
try:
    import pyvista as pv
except ImportError:
    print("\n[ERROR] PyVista is required for this native 3D desktop application.")
    print("        Please install it by running: pip install pyvista")
    print("        You may also need to run: pip install PyQt5  (or PySide6)")
    sys.exit(1)

def load_existing_labels():
    labels_file = 'data/engine_labels.json'
    if os.path.exists(labels_file):
        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to parse existing labels file: {e}")
    return []

if __name__ == '__main__':
    # Ensure working directory is the folder of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..', '..', 'public'))
    
    # Active box bounds tracking
    active_bounds = [-150, 150, -150, 150, -150, 150]
    selected_preset = 'compressor'
    saved_actors = []
    checkbox_widgets = []
    text_actors = []

    # Setup PyVista Plotter
    plotter = pv.Plotter(window_size=[1280, 720], title="Turbojet CAD 3D Bounding Box Editor")
    plotter.background_color = '#0d1117'
    
    # Add help overlay
    help_text = (
      "3D VIEWPORT CONTROLS:\n"
      "  · Rotate Camera: Left Click + Drag\n"
      "  · Zoom View: Scroll Wheel\n"
      "  · Pan View: Shift + Left Click + Drag\n\n"
      "BOX WIDGET CONTROLS:\n"
      "  · Drag center sphere to Translate\n"
      "  · Drag face/corner spheres to Scale\n\n"
      "HOTKEYS:\n"
      "  · Press [S] Key: Save active box widget to file\n"
      "  · Press [R] Key: Rename a saved box\n"
      "  · Press [D] Key: Delete active box key from database\n"
      "  · Press [Q] Key: Exit application\n\n"
      "Saved boxes are listed on the right.\n"
      "Click their checkbox to load them into the editor!"
    )
    plotter.add_text(help_text, position='upper_left', font_size=10, color='#6b7280')

    # Load GLTF model
    gltf_path = 'Turbojet Engine.gltf'
    if os.path.exists(gltf_path):
        print(f"[INFO] Loading CAD model: '{gltf_path}'...")
        try:
            dataset = pv.read(gltf_path)
            if isinstance(dataset, pv.MultiBlock):
                for i in range(len(dataset)):
                    block = dataset[i]
                    if block is not None and hasattr(block, "scale"):
                        block.scale([1.5, 1.5, 1.5], inplace=True)
            else:
                dataset.scale([1.5, 1.5, 1.5], inplace=True)
                
            plotter.add_mesh(dataset, color='#5fb3d9', opacity=0.7, ambient=0.2, diffuse=0.8, specular=0.5)
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load GLTF: {e}")
            sys.exit(1)
    else:
        print(f"[WARNING] GLTF model '{gltf_path}' not found at root.")
        print("[INFO] Activating fallback placeholder cylinder...")
        cylinder = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0), radius=150, height=800)
        plotter.add_mesh(cylinder, color='#5fb3d9', style='wireframe', line_width=1)

    # Function to redraw saved overlays in PyVista viewport
    def refresh_viewport_overlays():
        global saved_actors, text_actors
        # Clear previous actors
        for actor in saved_actors:
            plotter.remove_actor(actor)
        saved_actors.clear()
        
        for actor in text_actors:
            plotter.remove_actor(actor)
        text_actors.clear()
        
        labels = load_existing_labels()
        for item in labels:
            cx, cy, cz = item['center']
            w, h, d = item['size']
            key = item['key']
            
            box_mesh = pv.Box(bounds=(cx - w/2, cx + w/2, cy - h/2, cy + h/2, cz - d/2, cz + d/2))
            actor1 = plotter.add_mesh(box_mesh, style='wireframe', color='#38bdf8', line_width=2)
            actor2 = plotter.add_point_labels([[cx, cy, cz]], [key], font_size=10, text_color='#38bdf8')
            saved_actors.append(actor1)
            saved_actors.append(actor2)

    # Interactive click handler for viewport checkboxes
    def make_checkbox_callback(target_key):
        def callback(state):
            if not state:
                return
            
            # Reset all other checkboxes visually
            for key, cb in checkbox_widgets:
                if key != target_key:
                    cb.SetState(0)
            
            # Load bounds of selected key into active box widget
            labels = load_existing_labels()
            for item in labels:
                if item['key'] == target_key:
                    cx, cy, cz = item['center']
                    w, h, d = item['size']
                    xmin, xmax = cx - w/2, cx + w/2
                    ymin, ymax = cy - h/2, cy + h/2
                    zmin, zmax = cz - d/2, cz + d/2
                    
                    global active_bounds, selected_preset
                    active_bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
                    selected_preset = target_key
                    
                    # Set bounds directly on active widget representation
                    if hasattr(plotter, 'box_widget') and plotter.box_widget is not None:
                        rep = plotter.box_widget.GetRepresentation()
                        if rep is not None:
                            rep.SetBounds(xmin, xmax, ymin, ymax, zmin, zmax)
                            plotter.render()
                    print(f"[INFO] Loaded bounds for '{target_key}' into editor.")
                    break
        return callback

    # Re-draw the clickable checkbox widgets list inside the 3D viewport
    def refresh_viewport_checkboxes():
        global checkbox_widgets, text_actors
        # Clear previous widget overlays
        for key, cb in checkbox_widgets:
            cb.Off()
        checkbox_widgets.clear()
        
        labels = load_existing_labels()
        
        # Add side panel header text
        header = plotter.add_text("SAVED BOXES LIST:", position=(1050, 680), font_size=10, color='white')
        text_actors.append(header)
        
        # Position offset: list items down from the top-right
        start_y = 640
        for idx, item in enumerate(labels):
            key = item['key']
            y_pos = start_y - (idx * 32)
            
            # Add label text next to the checkbox
            lbl = plotter.add_text(f"{idx+1:02d}. {key}", position=(1085, y_pos), font_size=9, color='#e8b34c')
            text_actors.append(lbl)
            
            # Add interactive checkbox button
            cb = plotter.add_checkbox_button_widget(
                make_checkbox_callback(key),
                value=False,
                position=(1050, y_pos - 4),
                size=18,
                border_size=2,
                color_on='#059669',
                color_off='#374151',
                background_color='#1f2937'
            )
            checkbox_widgets.append((key, cb))

    # Box Widget Callback
    def box_callback(box_widget_dataset):
        global active_bounds
        active_bounds = list(box_widget_dataset.bounds)

    # Add Box Widget to Editor
    plotter.add_box_widget(
        box_callback, 
        bounds=active_bounds, 
        rotation_enabled=False,
        color='#f5d442'
    )

    # S key handler: prompts in temporary Tkinter modal dialog and saves box
    def save_box():
        xmin, xmax, ymin, ymax, zmin, zmax = active_bounds
        cx = int(round((xmin + xmax) / 2))
        cy = int(round((ymin + ymax) / 2))
        cz = int(round((zmin + zmax) / 2))
        w = int(round(xmax - xmin))
        h = int(round(ymax - ymin))
        d = int(round(zmax - zmin))
        
        import tkinter as tk
        from tkinter import simpledialog, messagebox
        
        # Open a temporary top-level modal dialog to prevent thread blocking
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        
        key = simpledialog.askstring(
            "Save Bounding Box", 
            f"Enter target component key (default: {selected_preset}):", 
            initialvalue=selected_preset, 
            parent=root
        )
        if not key:
            root.destroy()
            return
            
        desc = simpledialog.askstring(
            "Save Box Description", 
            f"Enter description notes for '{key}':", 
            parent=root
        )
        if desc is None:
            desc = ""
            
        labels_list = load_existing_labels()
        new_data = {
            "key": key.strip(),
            "description": desc.strip(),
            "center": [cx, cy, cz],
            "size": [w, h, d]
        }
        
        # Update or append
        updated = False
        for idx, item in enumerate(labels_list):
            if item.get('key') == new_data['key']:
                labels_list[idx] = new_data
                updated = True
                break
        if not updated:
            labels_list.append(new_data)
            
        try:
            with open('data/engine_labels.json', 'w', encoding='utf-8') as f:
                json.dump(labels_list, f, indent=2)
            
            # Refresh overlays and checkboxes inside viewport
            refresh_viewport_overlays()
            refresh_viewport_checkboxes()
            
            messagebox.showinfo("Success", f"Box label '{new_data['key']}' saved successfully!", parent=root)
            print(f"[SUCCESS] Saved label '{new_data['key']}' -> Center: {new_data['center']}, Size: {new_data['size']}")
        except Exception as err:
            messagebox.showerror("Error", f"Failed to save label: {err}", parent=root)
            print(f"[ERROR] Failed to save JSON database: {err}")
            
        root.destroy()

    # D key handler: prompts in temporary Tkinter modal dialog and deletes box
    def delete_box():
        labels_list = load_existing_labels()
        
        import tkinter as tk
        from tkinter import simpledialog, messagebox
        
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        
        if not labels_list:
            messagebox.showinfo("Info", "No boxes saved yet to delete.", parent=root)
            root.destroy()
            return
            
        choices = "\n".join([f"  · {item['key']}" for item in labels_list])
        key_to_delete = simpledialog.askstring(
            "Delete Box Label", 
            f"Enter the key of the box to delete:\n\nCurrently Saved Boxes:\n{choices}", 
            parent=root
        )
        
        if not key_to_delete:
            root.destroy()
            return
            
        key_to_delete = key_to_delete.strip()
        found = False
        for idx, item in enumerate(labels_list):
            if item.get('key').lower() == key_to_delete.lower():
                labels_list.pop(idx)
                found = True
                break
                
        if found:
            try:
                with open('data/engine_labels.json', 'w', encoding='utf-8') as f:
                    json.dump(labels_list, f, indent=2)
                
                # Refresh overlays and checkboxes inside viewport
                refresh_viewport_overlays()
                refresh_viewport_checkboxes()
                
                messagebox.showinfo("Success", f"Deleted box label '{key_to_delete}' successfully!", parent=root)
                print(f"[SUCCESS] Deleted box '{key_to_delete}' from database.")
            except Exception as err:
                messagebox.showerror("Error", f"Failed to delete label: {err}", parent=root)
        else:
            messagebox.showerror("Error", f"No saved box found with key '{key_to_delete}'!", parent=root)
        root.destroy()

    # R key handler: prompts in temporary Tkinter modal dialog and renames box
    def rename_box():
        global selected_preset
        labels_list = load_existing_labels()
        
        import tkinter as tk
        from tkinter import simpledialog, messagebox
        
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        
        if not labels_list:
            messagebox.showinfo("Info", "No boxes saved yet to rename.", parent=root)
            root.destroy()
            return
            
        choices = "\n".join([f"  · {item['key']}" for item in labels_list])
        
        # Determine the initial/default key to rename
        default_old = selected_preset if any(item['key'] == selected_preset for item in labels_list) else ""
        
        old_key = simpledialog.askstring(
            "Rename Bounding Box", 
            f"Enter the key of the box you want to rename:\n\nCurrently Saved Boxes:\n{choices}",
            initialvalue=default_old,
            parent=root
        )
        
        if not old_key:
            root.destroy()
            return
            
        old_key = old_key.strip()
        
        # Verify it exists
        target_item = None
        for item in labels_list:
            if item.get('key').lower() == old_key.lower():
                target_item = item
                break
                
        if not target_item:
            messagebox.showerror("Error", f"No saved box found with key '{old_key}'!", parent=root)
            root.destroy()
            return
            
        # Ask for the new key name
        new_key = simpledialog.askstring(
            "New Box Label", 
            f"Enter new key name for '{target_item['key']}':", 
            initialvalue=target_item['key'],
            parent=root
        )
        
        if not new_key:
            root.destroy()
            return
            
        new_key = new_key.strip()
        if not new_key:
            root.destroy()
            return
            
        # Check if new key already exists (and is different from old key)
        if new_key.lower() != target_item['key'].lower():
            if any(item.get('key').lower() == new_key.lower() for item in labels_list):
                messagebox.showerror("Error", f"A box with key '{new_key}' already exists!", parent=root)
                root.destroy()
                return
                
        # Perform rename
        old_name_actual = target_item['key']
        target_item['key'] = new_key
        
        # If we renamed the active preset, update it
        if selected_preset == old_name_actual:
            selected_preset = new_key
            
        try:
            with open('data/engine_labels.json', 'w', encoding='utf-8') as f:
                json.dump(labels_list, f, indent=2)
            
            # Refresh overlays and checkboxes inside viewport
            refresh_viewport_overlays()
            refresh_viewport_checkboxes()
            
            messagebox.showinfo("Success", f"Renamed box label '{old_name_actual}' to '{new_key}' successfully!", parent=root)
            print(f"[SUCCESS] Renamed box '{old_name_actual}' -> '{new_key}' in database.")
        except Exception as err:
            messagebox.showerror("Error", f"Failed to rename label: {err}", parent=root)
            
        root.destroy()

    # Bind S, R, and D keys
    plotter.add_key_event("s", save_box)
    plotter.add_key_event("r", rename_box)
    plotter.add_key_event("d", delete_box)

    # Initial overlays loading
    refresh_viewport_overlays()
    refresh_viewport_checkboxes()

    print("\n[INFO] Starting native 3D visualization editor window...")
    print("       Press [S] inside window to trigger terminal save prompts.")
    print("       Press [R] inside window to trigger terminal rename prompts.")
    print("       Press [D] inside window to trigger terminal delete prompts.")
    print("       Press [Q] inside window to exit.")
    plotter.show()
