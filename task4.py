# -*- coding: utf-8 -*-
'''
Моделирование отражения гармонического сигнала от слоя диэлектрика
'''

import math

import numpy as np
import numpy.typing as npt
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

from objects import LayerContinuous, LayerDiscrete, Probe

import boundary
import sources
import tools

from abc import ABCMeta, abstractmethod

class BoundaryBase(metaclass=ABCMeta):
    @abstractmethod
    def updateField(self, E, H):
        pass

class ABCFirstBase(BoundaryBase, metaclass=ABCMeta):
    '''
    Поглощающие граничные условия первой степени
    '''
    def __init__(self, eps, mu, Sc):
    
        temp = Sc / np.sqrt(mu * eps)
        self.koeffABC = (temp - 1) / (temp + 1)
        
        # E в предыдущий момент времени
        self.oldEz = np.zeros(1)
 
        
class ABCFirstLeft(ABCFirstBase):
    def updateField(self, E, H):
        E[0] = self.oldEz + self.koeffABC * (E[1] - E[0])
        self.oldEz = E[1]
       
       
class ABCFirstRight(ABCFirstBase):
    def updateField(self, E, H):       
        E[-1] = self.oldEz + self.koeffABC * (E[-2] - E[-1])
        self.oldEz = E[-2]

class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return math.floor(x / self.discrete + 0.5)


def sampleLayer(layer_cont: LayerContinuous, sampler: Sampler) -> LayerDiscrete:
    start_discrete = sampler.sample(layer_cont.xmin)
    end_discrete = (sampler.sample(layer_cont.xmax)
                    if layer_cont.xmax is not None
                    else None)
    return LayerDiscrete(start_discrete, end_discrete,
                         layer_cont.eps, layer_cont.mu, layer_cont.sigma)


def fillMedium(layer: LayerDiscrete,
               eps: npt.NDArray[float],
               mu: npt.NDArray[float],
               sigma: npt.NDArray[float]):
    if layer.xmax is not None:
        eps[layer.xmin: layer.xmax] = layer.eps
        mu[layer.xmin: layer.xmax] = layer.mu
        sigma[layer.xmin: layer.xmax] = layer.sigma
    else:
        eps[layer.xmin:] = layer.eps
        mu[layer.xmin:] = layer.mu
        sigma[layer.xmin:] = layer.sigma


if __name__ == '__main__':
    # Используемые константы
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi

    # Скорость света в вакууме
    c = 299792458.0

    # Электрическая постоянная
    eps0 = 8.854187817e-12

    #Fmin и Fmax
    Fmax = 5e9
    Fmin = 0

    # Дискрет по пространству в м
    dx = 5e-4
    
    # Число Куранта
    Sc = 1.0

    # Размер области моделирования в м
    maxSize_m = 2.2

    # Время расчета в секундах
    maxTime_s = 100e-9

    # Положение источника в м
    sourcePos_m = 0.8

    # Координаты датчиков для регистрации поля в м
    probesPos_m = [0.3]

    # Параметры слоев
    d0_m = 1.5
    eps_0 = 1

    d1_m = 0.6
    eps_1 = 2.5

    d2_m = 0.05
    eps_2 = 1.5

    eps_3 = 8.0
   
    layers_cont = [LayerContinuous(d0_m     , eps= eps_1, sigma=0.0),
                   LayerContinuous(d0_m+d1_m, eps= eps_2, sigma=0.0),
                   LayerContinuous(d0_m+d1_m+d2_m, eps= eps_3, sigma=0.0)
                   ]

    # Скорость обновления графика поля
    speed_refresh = 1000

    # Переход к дискретным отсчетам
    # Дискрет по времени
    dt = dx * Sc / c

    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # Время расчета в отсчетах
    maxTime = sampler_t.sample(maxTime_s)

    # Дискрет по частоте
    df = 1.0 / (maxTime * dt)

    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)

    # Положение источника в отсчетах
    sourcePos = sampler_x.sample(sourcePos_m)

    layers = [sampleLayer(layer, sampler_x) for layer in layers_cont]

    # Датчики для регистрации поля
    probesPos = [sampler_x.sample(pos) for pos in probesPos_m]
    probes = [Probe(pos, maxTime) for pos in probesPos]

    # Массив хранящий значения падающей волны
    source_E = np.ones(maxTime)

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)

    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)

    # Проводимость
    sigma = np.zeros(maxSize)

    for layer in layers:
        fillMedium(layer, eps, mu, sigma)

    # Коэффициенты для учета потерь
    loss = sigma * dt / (2 * eps * eps0)
    ceze = (1.0 - loss) / (1.0 + loss)
    cezh = W0 / (eps * (1.0 + loss))

    # Источник
    magnitude = 1.0
    dw=30
    dg=120
    
    source = sources.Gaussian(magnitude, dg, dw)

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)

    # Создание экземпляров классов граничных условий
    boundary_left = ABCFirstLeft(eps[0], mu[0], Sc)
    boundary_right = ABCFirstRight(eps[-1], mu[-1], Sc)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -2.1
    display_ymax = 2.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(dx, dt,
                                        maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel,
                                        title='Расчет зависимости\
 модуля коэффициента отражения от частоты.')

    display.activate()
    display.drawSources([sourcePos])
    display.drawProbes(probesPos)
    for layer in layers:
        display.drawBoundary(layer.xmin)
        if layer.xmax is not None:
            display.drawBoundary(layer.xmax)

    for t in range(1, maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] += -1/W0*source.getE(t-1)

        # Расчет компоненты поля E
        Ez[1:-1] = ceze[1: -1] * Ez[1: -1] + cezh[1: -1] * (Hy[1:] - Hy[: -1])

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        source_E[t] = source.getE(t)
        Ez[sourcePos] += source.getE(t)

        boundary_left.updateField(Ez, Hy)
        boundary_right.updateField(Ez, Hy)

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % speed_refresh == 0:
            display.updateData(display_field, t)

    display.stop()
    
    # Расчет спектров излученного и отрженного сигналов
    Ez1Spec = fftshift(np.abs(fft(probe.E)))
    Ez0Spec = fftshift(np.abs(fft(source_E)))

    Gamma = Ez1Spec / Ez0Spec

    tlist = np.arange(0, maxTime * dt, dt)
    flist = np.arange(-maxTime / 2 , maxTime / 2 , 1)*df
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set_xlim(0, 2e-8)
    ax1.set_ylim(-0.5, 1.2)
    ax1.set_xlabel('t, с')
    ax1.set_ylabel('Ez, В/м')
    ax1.plot(tlist, source_E)
    ax1.plot(tlist, probe.E)
    ax1.legend(['Падающий сигнал',
     'Отраженный сигнал'],
     loc='lower right')
    ax1.minorticks_on()
    ax1.grid()

    ax2.set_xlim(Fmin,Fmax)
    ax2.set_xlabel('f, Гц')
    ax2.set_ylabel('|F{Ez}|, В*с/м')
    ax2.plot(flist, Ez0Spec)
    ax2.plot(flist, Ez1Spec)
    ax2.legend(['Спектр падающего сигнала',
     'Спектр отраженного сигнала'],
     loc='upper right')
    ax2.minorticks_on()
    ax2.grid()

    ax3.set_xlim(Fmin, Fmax)
    ax3.set_ylim(0, 0.65)
    ax3.set_xlabel('f, Гц')
    ax3.set_ylabel('|Г|, б/р')
    ax3.plot(flist, Gamma)
    ax3.minorticks_on()
    ax3.grid()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
