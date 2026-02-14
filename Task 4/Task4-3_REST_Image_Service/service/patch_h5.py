import shutil, json, h5py

src = r"..\model\vehicle_classifier_mobilenetv2.h5"
dst = r"..\model\vehicle_classifier_mobilenetv2_patched.h5"

shutil.copy2(src, dst)

with h5py.File(dst, "r+") as f:
    mc = f.attrs.get("model_config")
    if mc is None:
        raise RuntimeError("model_config not found in H5 file")
    if isinstance(mc, (bytes, bytearray)):
        mc = mc.decode("utf-8")

    data = json.loads(mc)
    s = json.dumps(data)
    s2 = s.replace('"batch_shape"', '"batch_input_shape"')

    if s == s2:
        print("No batch_shape found. Nothing changed.")
    else:
        f.attrs.modify("model_config", s2.encode("utf-8"))
        print("Patched: batch_shape -> batch_input_shape")

print("Wrote:", dst)