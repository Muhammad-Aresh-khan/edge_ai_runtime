# SafeChild: Setup aur Live Pipeline

# 1. Setup (sirf ek dafa, backend pe)

### Room ki photo
- Pehle khali room ki photo di jayegi, jo setup pipeline mein jayegi.
- Photo wohi honi chahiye jo camera ki final position se li gayi ho, aur bacha, pets ya koi insaan frame mein na ho.
- Isi photo ko "reference photo" ke taur pe save bhi karna hai, kyunke drift check ko isi ki zaroorat hai.

### Danger zones ki detection
Setup pipeline ke andar ye chalte hain:
- **YOLO Hazard Detection:** jo hamara apna model hai.
- **YOLO-World:** jo open vocabulary object detection model hai.
- **Optional lightweight VLM:** in ke sath ham ise bhi chala sakte hain.

In 3 models se room mein jitne bhi danger zones hain wo mark ho jayenge.

### Parent review
- Parents review karenge. Agar koi particular danger zone hatana hai to hata sakte hain.
- Agar koi danger zone miss ho gaya to parents ko ye option bhi di jayegi ke wo danger zone khud mark kar lein.
- Khud mark karne ka matlab hai app ki screen pe polygon draw karna (ungli se corners tap karke) aur hazard ki type select karna.
- Review ki screen app mein hogi. Backend sirf AI processing karega aur zones save karega.

### Zones save aur sync
- Review ke baad final zones (polygons ki JSON) camera ID ke saath save hote hain, aur edge device pe sync hote hain.
- Isi se setup aur live pipeline jurti hain.
- Edge un zones ko local cache mein rakhta hai, taake internet na ho tab bhi system chalta rahe.

### Bache ka naam
- Bache ka naam bhi setup mein daalna hai, taake alert mein naam le sake.

### Setup dobara kab chalega
**YE SARA SETUP HAI JO BAS EK DAFA CHALEGA.** Dobara sirf tab chalega jab:
- camera hil jaye,
- furniture ya kamra badle,
- ya naya camera lage.

---

# 2. Live Pipeline (edge pe, continuously)

Ye pipeline edge pe continuously chalegi, is order mein:
1. **Toddler detection** model
2. **YOLO pose estimation** model
3. **Geometry maths**
4. **Voice alerts**

### Geometry aur pose
- Geometry mein bache ke paon aur body ko saved zones se compare kiya jata hai, aur result **DANGER, WARNING, SAFE ya IDLE** nikalta hai.
- Pose se pata chalta hai ke bacha chadh raha hai, jhuk raha hai ya haath barha raha hai, aur ye verdict ko adjust karta hai.

### Alerts
- Voice ke sath screen pe HUD (red, amber, green boxes) aur danger ka snapshot bhi save hota hai.
- Ye sab edge pe local hota hai, internet ka intezar nahi.

---

# 3. Support Layer (optional)

Us ke baad ek support layer bhi add kar sakte hain.

### Drift check
- **Matlab:** agar camera thora sa hil gaya to wo purane coordinates ko new coordinates se match karke dobara danger zones ki actual position nikalega. 
- **Tareeka:** setup ki reference photo aur naye frame mein common points dhoondo (ORB features), phir homography se shift nikalo, aur wahi shift zones ke polygons pe lagao.
- **Chhota shift:** zones khud adjust ho jayein.
- **Bohot zyada shift:** auto-correct na karo, balkay parent ko dobara setup karne ka alert do.
- Ye check tab chalao jab frame mein bacha na ho.

### Support VLM
- Sath mein ek support ke liye VLM chala sakte hain, lekin privacy issue se bachne ke liye ham bache ke face ko blur kar sakte hain, jaise mobile mein mosaic filter hota hai.
- Ye layer by default band rahe, sirf parent ki ijazat se on ho.
- Sirf shak wale frames pe chale, aur frame crop aur downscale karke bheja jaye.
- **Timeout rakho:** agar VLM jawab na de ya internet na ho to geometry aur pose ka verdict hi final rahe. Alert kabhi VLM ka intezar na kare.