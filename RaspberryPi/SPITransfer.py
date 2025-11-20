#SPI bus number
SPI_bus = 0

#chip select pin. Can be set high(1) or low(0)
CS_pin = 1

#enables SPI
spi = spidev.SpiDev()

#opens connection to bus and peripheral
spi.open(bus, CS_pin)

spi.max_speed_hz = 10000
spi.mode = 0

message = "Hello World! This is the Raspberry Pi!"

spi.xfer(message)
