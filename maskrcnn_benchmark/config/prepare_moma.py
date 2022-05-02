from paths_catalog import DatasetCatalog
from momaapi import MOMA


    
def create_dataset(moma, split, classes):
    ids_hoi = moma.get_ids_hoi(split=split)
    anns_hoi = moma.get_anns_hoi(ids_hoi)
    ids_act = moma.get_ids_act(ids_hoi=ids_hoi)
    print("IDS ACT:", ids_act)
    metadata = moma.get_metadata(ids_act=ids_act)
    image_paths = moma.get_paths(ids_hoi=ids_hoi)
    print(len(ids_hoi))
    dataset_dicts = []
    for ann_hoi, image_path, metadatum in zip(anns_hoi, image_paths, metadata):
        record = {}
        record["file_name"] = image_path
        record["image_id"] = ann_hoi.id
        record["height"]= metadatum.height
        record["width"] = metadatum.width

        objs = []

        for actor in ann_hoi.actors:
            bbox = actor.bbox
            id = actor.id
            actor_cname = actor.cname
            if actor_cname in classes:
                class_id = classes.index(actor_cname)
                obj = {
                    "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": class_id,
                }
                if actor_cname == "crowd":
                    obj["iscrowd"] = 1 # Set iscrowd to be true for crowds
                objs.append(obj)
            # else:
            #     print("actor_cname has 0 instance:", actor_cname)

        for object in ann_hoi.objects:
            bbox = object.bbox
            id = object.id
            object_cname = object.cname
            object_cid = object.cid + len(moma.taxonomy['actor'])
            if object_cname in classes:
                class_id = classes.index(object_cname)
                obj = {
                    "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": class_id,
                }
                objs.append(obj)
            # else:
            #     print("object_cname has 0 instance:", object_cname)

        record["annotations"] = objs
        # print("added", objs)
        dataset_dicts.append(record)

    return dataset_dicts
    

def register_dataset(path_to_moma='../../data/moma/'):
    print("Loading MOMA API...")
    moma = MOMA(path_to_moma)
    print("MOMA API Loaded!")
    
    print("Registering MOMA dataset")
    num_instances_threshold = 50
    classes = moma.get_cnames(concept='actor', threshold=num_instances_threshold, split='either') # Only train on actors and objects from val set
    classes += moma.get_cnames(concept='object', threshold=num_instances_threshold, split='either') # Ensure there are at least 50 examples to include

    
    for split in ["train", "val"]:
        print("Registering " + split + " dataset...")
        print(create_dataset(moma, split, classes))
        # DatasetCatalog.register("moma_" + split, lambda split=split: create_dataset(moma, split, classes))
    print("MOMA dataset registered!")
    return
    

# For testing purposes
register_dataset("/home/zaned/sgg/Scene-Graph-Benchmark.pytorch/datasets/moma/")
data_dict = DatasetCatalog.get("moma_train")
print(data_dict)