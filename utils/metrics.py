import torch

def process_clicks(y: torch.Tensor, y_hat: torch.Tensor):
    assert len(y) == len(y_hat)

    total_true_clicks = 0
    total_false_clicks = 0
    total_missed_clicks = 0
    total_detected_clicks = 0
    on_set_offsets = []
    off_set_offsets = []
    drops = []

    def process_clicks_generator():
        nonlocal total_true_clicks, total_false_clicks, total_missed_clicks, total_detected_clicks
        nonlocal on_set_offsets, off_set_offsets, drops

        def y_hat_activated(already_clicked):
            nonlocal total_true_clicks, total_false_clicks, total_missed_clicks, total_detected_clicks
            nonlocal on_set_offsets, off_set_offsets, drops
            # false click or early detection of clicks
            duration = 1
            while True:
                this_y, this_y_hat = yield
                # print("yield y_hat_activated")
                if this_y == 0 and this_y_hat == 0:
                    if already_clicked == 1:
                        off_set_offsets += [duration]
                    else:
                        total_false_clicks += 1
                    break
                elif this_y == 0 and this_y_hat == 1:
                    duration += 1
                    continue
                elif this_y == 1 and this_y_hat == 0:
                    if already_clicked == 1:
                        off_set_offsets += [duration / 2]
                        on_set_offsets += [-duration / 2]
                    else:
                        on_set_offsets += [-duration]
                    yield from y_activated(already_detected=1, already_dropped=1)
                    break
                elif this_y == 1 and this_y_hat == 1:
                    if already_clicked == 1:
                        off_set_offsets += [duration / 2]
                        on_set_offsets += [-duration / 2]
                    else:
                        on_set_offsets += [-duration]
                    yield from y_activated(already_detected=1, already_dropped=0)
                    break

        def y_activated(already_detected, already_dropped):
            nonlocal total_true_clicks, total_false_clicks, total_missed_clicks, total_detected_clicks
            nonlocal on_set_offsets, off_set_offsets, drops
            dropped_duration = 0 if not already_dropped else 1
            while True:
                this_y, this_y_hat = yield
                # print("yield y_activated")
                if this_y == 0 and this_y_hat == 0:
                    total_true_clicks += 1
                    if already_detected:
                        if dropped_duration != 0:
                            off_set_offsets += [-dropped_duration]
                        else:
                            off_set_offsets += [0]
                        total_detected_clicks += 1
                    else:
                        total_missed_clicks += 1
                    break
                elif this_y == 0 and this_y_hat == 1:
                    total_true_clicks += 1
                    if already_detected:
                        total_detected_clicks += 1
                        if dropped_duration != 0:
                            drops += [dropped_duration]
                        yield from y_hat_activated(1)
                    else:
                        total_missed_clicks += 1
                        yield from y_hat_activated(0)
                    break
                elif this_y == 1 and this_y_hat == 0:
                    dropped_duration += 1
                    continue
                elif this_y == 1 and this_y_hat == 1:
                    if already_detected == 0:
                        already_detected = 1
                        on_set_offsets += [dropped_duration]
                        dropped_duration = 0
                    if dropped_duration != 0:
                        drops += [dropped_duration]
                    dropped_duration = 0
                    continue

        while True:
            # nothing happened
            this_y, this_y_hat = yield
            # print("yield default")
            if this_y == 0 and this_y_hat == 0:
                continue
            elif this_y == 0 and this_y_hat == 1:
                yield from y_hat_activated(already_clicked=0)
            elif this_y == 1 and this_y_hat == 0:
                yield from y_activated(already_detected=0, already_dropped=1)
            elif this_y == 1 and this_y_hat == 1:
                on_set_offsets += [0]
                yield from y_activated(already_detected=1, already_dropped=0)

    process_gen = process_clicks_generator()
    next(process_gen)
    for i in range(len(y)):
        # print(f"{i}:")
        process_gen.send((y[i], y_hat[i]))
    process_gen.send((0, 0))
    return \
        total_true_clicks, total_false_clicks, total_missed_clicks, total_detected_clicks,\
        on_set_offsets, off_set_offsets, drops


