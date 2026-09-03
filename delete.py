from sipyco.pc_rpc import Client

slm_cam = Client("127.0.0.1", 4000, target_name="slm_cam")

atoms = Client("127.0.0.1", 4000, target_name="atoms")

atoms.load_atoms(400)

slm_cam.identify_filled_atom_sites()

slm_cam.close_rpc()