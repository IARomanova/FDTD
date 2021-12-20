import numpy
import math
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

import tools

class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return math.floor(x / self.discrete + 0.5)

class GaussianModPlaneWave:
    ''' Класс с уравнением плоской волны для модулированного гауссова сигнала в дискретном виде
    d - определяет задержку сигнала.
    w - определяет ширину сигнала.
    Nl - количество ячеек на длину волны.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    mu - относительная магнитная проницаемость среды, в которой расположен источник.
    '''

    def __init__(self, d, w, Nl, Sc=1.0, eps=1.0, mu=1.0):
        self.d = d
        self.w = w
        self.Nl = Nl
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        '''
        return (numpy.sin(2 * numpy.pi / self.Nl * (q * self.Sc - m * numpy.sqrt(self.eps * self.mu))) *
                numpy.exp(-(((q - m * numpy.sqrt(self.eps * self.mu) / self.Sc) - self.d) / self.w) ** 2))

if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Скорость света в вакууме
    c = 299792458.0

    # Число Куранта
    Sc = 1.0

    # Время расчета в секундах
    maxTime_s = 250e-9
    
    
    # Размер области моделирования в метрах
    maxSize_m = 4.0

    # Дискрет по пространству в м
    dx = 1e-3

    # Скорость обновления графика поля
    speed_refresh = 500

    # Параметры среды
    # Диэлектрическая проницаемость
    eps=9.0

    # Магнитная проницаемость
    mu=1

    # Скорость распространения волны
    v=c/numpy.sqrt(eps)
    
    # Переход к дискретным отсчетам
    # Дискрет по времени
    dt = dx * Sc / v

    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # Время расчета в отсчетах
    maxTime = sampler_t.sample(maxTime_s)

    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)
    
    # Положение источника в метрах
    sourcePos_m = maxSize_m/2
    # Положение источника в отсчетах
    sourcePos = math.floor(sourcePos_m / dx + 0.5) 

    # Положение датчика
    probesPos_m = 1
    # Датчики для регистрации поля
    probesPos = [math.floor( probesPos_m / dx + 0.5)]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    probesPos_1 = math.floor( probesPos_m / dx + 0.5)

    # Массивы для Ez и Hy
    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)
    source = GaussianModPlaneWave(3000,400,160, Sc, eps, mu)

    # Коэффициенты для расчета ABC второй степени
    # Sc' для левой границы
    Sc1Left = Sc / numpy.sqrt(mu * eps)

    k1Left = -1 / (1 / Sc1Left + 2 + Sc1Left)
    k2Left = 1 / Sc1Left - 2 + Sc1Left
    k3Left = 2 * (Sc1Left - 1 / Sc1Left)
    k4Left = 4 * (1 / Sc1Left + Sc1Left)

    # Ez[0: 2] в предыдущий момент времени (q)
    oldEzLeft1 = numpy.zeros(3)

    # Ez[0: 2] в пред-предыдущий момент времени (q - 1)
    oldEzLeft2 = numpy.zeros(3)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1
    
    display = tools.AnimateFieldDisplay(dx, dt,
                                        maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for t in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)
        # Граничные условия PMC (справа)
        Hy[-1]=0

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu) * source.getE(0, t)

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1: -1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps * mu)) *
                          source.getE(-0.5, t + 0.5))
        # Граничные условия ABC второй степени (слева)
        Ez[-1]=0
        Ez[0] = (k1Left * (k2Left * (Ez[2] + oldEzLeft2[0]) +
                           k3Left * (oldEzLeft1[0] + oldEzLeft1[2] - Ez[1] - oldEzLeft2[1]) -
                           k4Left * oldEzLeft1[1]) - oldEzLeft2[2])

        oldEzLeft2[:] = oldEzLeft1[:]
        oldEzLeft1[:] = Ez[0: 3]
        
        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % speed_refresh == 0:
            display.updateData(display_field, t)

    display.stop()

    # Расчёт спектра сигнала
    EzSpec = fftshift(numpy.abs(fft(probe.E)))
   
    # Рассчёт шага частоты
    df = 1.0 / (maxTime * dt)
    # Рассчёт частотной сетки
    freq = numpy.arange(-maxTime / 2 , maxTime / 2 )*df
    # Оформляем сетку
    tlist = numpy.arange(0, maxTime * dt, dt) 

    # Вывод сигнала и спектра зарегестрированых в датчике
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_xlim(1.65e-7, 2e-7)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_xlabel('t, с')
    ax1.set_ylabel('Ez, В/м')
    ax1.plot(tlist, probe.E,lw=2)
    ax1.minorticks_on()
    ax1.grid()
    ax2.set_xlim(-4e9, 4e9)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel('f, Гц')
    ax2.set_ylabel('|S| / |Smax|, б/р')
    ax2.plot(freq, EzSpec / numpy.max(EzSpec),lw=2)
    ax2.minorticks_on()
    ax2.grid()
    plt.show()
