import aifourier as aif
import matplotlib.pyplot as plt

# analyze audio
df = aif.analyze("examples/bird.mp3", max_modes=128, epochs=128, learning_rate=0.0001)

print(df.head())

# plot amplitude vs frequency
plt.figure(figsize=(10,5))
plt.plot(df["Frequencies"], df["Amplitudes"], 'o')
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("AI Fourier Spectrum")
plt.grid()
plt.savefig("examples/spectrum.png")
plt.show()

# save to csv
df.to_csv("examples/spectrum.csv", index=False)