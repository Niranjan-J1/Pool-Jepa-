# D-002: Camera

**Status:** IN PROGRESS. This is a walkthrough with you: step 1 is approved, step 2 is proposed.
**Date:** 2026-11-23
**Inputs:** the estimated Soozier 38" playfield, 86 × 43 cm. With a 3 cm margin each side, the camera must see 0.92 × 0.49 m.

## Step 1: Requirements (APPROVED 2026-11-23)
| ID | Requirement | Why (source) |
|---|---|---|
| C-1 | Frame rate **30 fps**. Retrain on frame pairs 2 sim steps apart in the re-target step. | The sim runs at 60 Hz (`DT`). Your call: changing the training data is cheaper than buying a 60 fps camera. |
| C-2 | Exposure **lockable, 1/250 s or shorter; 1/120 s acceptable** | Planning reads only still frames (before the strike, and after everything settles), so blur matters only mid-shot. 1/250 s gives 1 cm of blur at 2.5 m/s, about the encoder's own error. |
| C-3 | Focus and white balance **lockable or fixed** | Background subtraction assumes a stable image. |
| C-4 | Resolution **720p or better** | 720p over 0.92 m is 1,391 px/m (target ≥600 after straightening); a 1.25" ball is about 44 px. |
| C-5 | Field of view: covers 0.92 m from a **reachable height of about 0.5–1.0 m** | 70° FOV → 0.66 m; 60° → 0.80 m; 50° → about 1.0 m. |
| C-6 | Camera-to-laptop delay **≤100 ms** | The robot moves 3 cm in 100 ms at 0.3 m/s (Stage 1 alignment). |
| C-7 | A **flicker-free bright lamp** is part of the camera system | Short exposure needs light; cheap LEDs flicker at 120 Hz and cause banding. |
| C-8 | A rigid mount, optical axis within about 5° of vertical | Keeps the homography valid between sessions (`docs/camera_contract.md`). |

## Step 2: Candidates (PROPOSED)
| | A: your phone as a webcam | B: Logitech C270 (new) | C: used 1080p webcam (e.g. C920) |
|---|---|---|---|
| Cost | **C$0** | C$28–50 ([Canadian Tire C$39.99](https://www.canadiantire.ca/en/pdp/logitech-c270-usb-720p-hd-webcam-with-built-in-mic-0695022p.html), [Walmart.ca ~C$28](https://www.walmart.ca/en/c/kp/logitech-camera), [UW store C$49.99](https://wstore.uwaterloo.ca/products/01220-logitech-c270-webcam)) | about C$25–40, ESTIMATE (marketplace) |
| Resolution / fps | 720p–1080p at 30 fps baseline via Iriun/DroidCam; 60 fps not guaranteed ([Iriun guide](https://iriun.github.io/obs-setup.html), [DroidCam forum](https://obsproject.com/forum/threads/droidcam-obs-camera.145220/post-702472)) | 720p at 30 | 1080p at 30 |
| C-2 exposure lock | the app or the phone's camera settings; Iriun itself advises locking it ([guide](https://iriun.github.io/obs-setup.html)). **Must test.** | **Unverified**: needs a UVC exposure-control test | usually yes on C920-class cameras; **must test** |
| C-3 focus | lockable in the app | **fixed focus** ([Best Buy spec](https://www.bestbuy.ca/en-ca/product/logitech-c270-hd-webcam/19460921)), which is good: it can't drift | autofocus, which can be switched off via UVC |
| C-5 FOV | about 70–80° main camera, ESTIMATE → about 0.6 m height | narrower, about 50–55°, ESTIMATE → about 1.0 m height | about 70° |
| C-6 delay | USB cable is lowest; Wi-Fi is worse ([Iriun](https://iriun.github.io/)). **Must measure.** | direct UVC, typically low | direct UVC, typically low |
| Downsides | the phone is tied up during sessions; a phone holder is needed; it depends on an app and a driver | might not lock exposure; lowest quality | used condition; must be tested before paying |

**Not counted before, flagged:** any camera needs a **mount** (an arm or clamp, about C$15–25, or a shelf you already have) and a **lamp** (about C$15–20, or one you own). Rough total then:
- table C$94;
- robot about C$70–90;
- mount and lamp C$0–45;
- camera C$0–40.

That's about **C$164–269**, so the camera path matters for staying under C$200.
