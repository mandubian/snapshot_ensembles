from tensorboard.backend.event_processing import event_accumulator

def tensorboard_event_accumulator(
    file,
    loaded_scalars=0, # load all scalars by default
    loaded_images=4, # load 4 images by default
    loaded_compressed_histograms=500, # load one histogram by default
    loaded_histograms=1, # load one histogram by default
    loaded_audio=4, # loads 4 audio by default
):
    ea = event_accumulator.EventAccumulator(
        file,
        size_guidance={ # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: loaded_compressed_histograms,
            event_accumulator.IMAGES: loaded_images,
            event_accumulator.AUDIO: loaded_audio,
            event_accumulator.SCALARS: loaded_scalars,
            event_accumulator.HISTOGRAMS: loaded_histograms,
        }
    )
    ea.Reload()
    return ea
