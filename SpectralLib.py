from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class SpectralLib():

    def __init__(self, data_path: str, schema_path: str, remove_absorpton: list[tuple] = False, **kwargs) -> None:
        '''
        Create an instance of `SpectralLib`. The path to the data (*.lib) and the schema (*.HDR) must be provided.
        '''
        keys = list()
        values = list()
        for k, v in kwargs.items():
            keys.append(k)
            values.append(v)
                
        if (('empty' in keys) & (True in values)):
            pass

        else:
            self._sp = schema_path
            self._dp = data_path
            self._rm = remove_absorpton
            self._s  = self._get_schema(self._sp)
            self._d  = self._get_data(self._dp)
            self._t  = self._get_stat()
            self._m  = '+1'
            self.s1_prop = np.NAN
            self.s2_prop = np.NAN
            self.s3_prop = np.NAN

    def _get_data(self, data_path: str) -> pd.DataFrame:
        '''
        Read the ENVI spectral library file (*.lib) and turn it into a `pandas.DataFrame`.
        '''
        with open(data_path, 'rb') as f:
            reflectance = np.fromfile(f, dtype = np.float64).reshape((self._s['lines'], self._s['samples'])).T  # Convert to a rotated matrix than tilt it back
        
        data = pd.DataFrame(reflectance, columns = self._s['spectra names'], index = self._s['wavelength'])
        data[data > 1] = np.NAN

        return data

    def _get_schema(self, schema_path: str) -> dict:
        '''
        Read the ENVI spectral library schema file (*.HDR) and turn it into a `dict`.
        '''
        schema = dict()
        
        def to_number(x: str) -> float:
            '''
            Convert the string into numbers if possible, otherwise keep it the same.
            '''
            try:
                x = float(x)

                if(x % 1 == 0):
                    x = int(x)

            except:
                pass

            return x

        with open(schema_path) as f:

            texts = f.readlines()

        for row, line in enumerate(texts):
            if (line.find('=') != -1):
                title = line[0:line.find('=')].strip()

                if ((line.find('{', line.find('=')) == -1) | (line.find('}', line.find('='))) - (line.find('{', line.find('='))) > 0):
                    contents = to_number(line[line.find('=')+1:].strip())

                else:
                    bracket_open = True
                    stepper     = row

                    while (bracket_open):
                        if (texts[stepper].find('}') != -1):
                            bracket_open = False
                            contents = [l.split(',') for l in texts[row:stepper+1]][1:]     # Remove title and split the nested list
                            contents = [x for xs in contents for x in xs]                   # Flatten the list
                            contents = list(filter(None, [i.strip() for i in contents]))    # Remove tailing space & empty string
                            contents[-1] = contents[-1][:-1]                                # Remove tailing right bracket
                            contents = [to_number(i) for i in contents]

                        else:
                            stepper += 1

                schema[title] = contents

        return schema
    
    def _get_stat(self) -> pd.DataFrame:
        '''
        Give the min, 25%, 50%, 75%, max, and geometric mean of the SpectralLib on each wavelength.
        '''
        stat = self._d.T.describe().iloc[1:, :].T
        geometric_mean = list()
        for i in range(len(self._d.index)):
            temp = 1
            nums = self._d.iloc[i, :].values.flatten().tolist()
            for j in nums:
                if ((j > 0) & (j <= 1) & (j != np.NAN)):
                    temp = temp * j
            
            geometric_mean.append(temp ** (1 / len(self._d.columns)))
        
        stat['geomean'] = geometric_mean
        return stat

    def __getitem__(self, index) -> pd.Series:
        return self._d[index]

    def __len__(self) -> int:
        return self._d.shape[1]

    def __repr__(self) -> str:
        return f'SpectralLib [{self._s["wavelength"][0]} - {self._s["wavelength"][-1]} {self._s["wavelength units"]}] ({self._d.shape[1]} spectra x {self._d.shape[0]} samples)'

    def apply(self, action: dict) -> 'SpectralLib':
        '''
        Apply function to the `SpectralLib`.
        '''
        temp = self.copy()
        temp.data = temp._d.apply(**action)
        return temp

    def collection(*args, **kwargs) -> 'SpectralLibCollection':
        '''
        Make a spectral library collection for use together.
        
        Examples
        --------
        One should always assign the name to the spectral libraries, such as:
        ```
        >>> water_spectral = SpectralLib('Water.lib', 'Water.HDR')
        >>> SpectralLib.collection(water_spectral)   # Anonymous, bad practice
        >>> SpectralLib.collection([water_spectral]) # Anonymous, bad practice
        >>> SpectralLib.collection({'water': water_spectral})
        >>> SpectralLib.collection(water = water_spectral)
        
        SpectralLibCollection of 1 SpectralLib: [Index 0]   # Not informative
        SpectralLibCollection of 1 SpectralLib: [Index 0]
        SpectralLibCollection of 1 SpectralLib: [water]
        SpectralLibCollection of 1 SpectralLib: [water]
        ```
        '''
        collection = dict()
        for i, arg in enumerate(args):
            if isinstance(arg, SpectralLib):
                collection[f'Index {i}'] = arg
            
            elif isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, SpectralLib):
                        collection[k] = v
                    
                    else:
                        raise ValueError('Only SpectralLib is accepted in the arguments')
            elif isinstance(arg, list):
                for j, item in enumerate(arg):
                    if isinstance(item, SpectralLib):
                        collection[f'Index {i}-{j}'] = item

                    else:
                        raise ValueError('Only SpectralLib is accepted in the arguments')

            else:
                raise ValueError('Only SpectralLib is accepted in the arguments')

        for k, v in kwargs.items():
            if isinstance(v, SpectralLib):
                collection[k] = v
            
            else:
                raise ValueError('Only SpectralLib is accepted in the arguments')
            
        return SpectralLibCollection(collection)
    
    def copy(self) -> 'SpectralLib':
        '''
        Copy the current `SpectralLib`.
        '''
        temp = SpectralLib(0, 0, empty = True)
        temp._d  = self._d
        temp._dp = self._dp
        temp._m  = self._m
        temp._rm = self._rm
        temp._s  = self._s
        temp.s1_prop = self.s1_prop
        temp.s2_prop = self.s2_prop
        temp.s3_prop = self.s3_prop
        temp._sp = self._sp
        temp._t  = self._t
        return temp

    def differential(self, nth: int = 1) -> 'SpectralLib':
        '''
        Calculate the differential of the original spectra.
        '''
        diff = self.copy()
        data = pd.DataFrame()
        remove = list()
        for start, end in diff._rm:
            diff._d = diff._d[(diff._d.index <= start)|(diff._d.index >= end)]
            remove.append(np.arange(start, end + 1).tolist())

        remove = [x for xs in remove for x in xs]

        for i in range(nth):
            for j in range(len(diff._d.index)):
                c = 1
                while (j + c < len(diff._d.index)):
                    if (diff._d.index[j + c] not in remove):
                        data = pd.concat([data, pd.DataFrame((diff._d.iloc[j + c] - diff._d.iloc[j]) / c).T.set_index(pd.Index([diff._d.index[j]]))])
                        break

                    else:
                         c += 1
                
                if (j == len(diff._d.index) - 1):
                    data = pd.concat([data, data.iloc[-1].T.set_index(pd.Index([diff._d.index[j]]))])
            
            diff._d = data
            data = pd.DataFrame()
        
        diff.data = diff._d
        diff._m = '-1'
        return diff

    def low_pass_filter(self, gap: int) -> 'SpectralLib':
        '''
        Apply a low-pass filter to y-axis over x-axis, accoording to the `gap` on x-axis. (Basically a moving average filter.)
        '''
        filtered = self.copy()
        moving_average = list()
        for start, end in filtered._rm:
            filtered._d = filtered._d[(filtered._d.index <= start)|(filtered._d.index >= end)]

        for i in filtered._d.index:
            data = filtered._d[(filtered._d.index >= i) & (filtered._d.index <= i + gap)]
            missing_count = gap - len(data)

            if (missing_count > 0):
                test = data.loc[np.repeat(data.index[-1], missing_count)]
                data = pd.concat([data, test])

            moving_average.append(pd.DataFrame(data.mean()).rename(columns = {0: i}).T)

        filtered.data = pd.concat(moving_average)
        return filtered

    def stat_plots(self, name: str, primary_color: str = 'tab:blue', secondary_color: str = 'grey', ylim: tuple = None, title: str = None) -> None:
        '''
        Give the plot according to `SpectralLib.stat`.
        '''
        data = self._t
        fig, ax = plt.subplots()
        if (self._m == '+1'):
            ax.set(xlabel = 'Wavelength (nm)', ylabel = 'Reflectance', ylim = (0, 0.5), title = f'{name} Spectra Distibution')

        elif (self._m == '-1'):
            ax.set(xlabel = 'Wavelength (nm)', ylabel = 'Slope of Reflectance', ylim = (-0.01, 0.01), title = f'{name} Spectra Slope Distibution')

        if ylim:
            ax.set(ylim = ylim)

        if title:
            ax.set(title = title)

        if isinstance(self._rm, list):
            first_index = data.index.min()
            last_index = data.index.max()
            first_plot = True
            for i, itv in enumerate(self._rm):
                if isinstance(itv, tuple):
                        if (first_index <= min(itv)):
                            sub_data = data[(data.index >= first_index)&(data.index <= min(itv))]

                            if (first_plot):
                                ax.plot(sub_data.index, sub_data['mean'], color = secondary_color, linestyle = 'dashed', linewidth = 1, label = 'Mean')
                                ax.plot(sub_data.index, sub_data['50%'], color = primary_color, linewidth = 1.5, label = 'Median')
                                ax.fill_between(sub_data.index, sub_data['25%'], sub_data['75%'], facecolor = primary_color, alpha = 0.3, label = 'IQR Range')
                                first_plot = False

                            else:
                                ax.plot(sub_data.index, sub_data['mean'], color = secondary_color, linestyle = 'dashed', linewidth = 1)
                                ax.plot(sub_data.index, sub_data['50%'], color = primary_color, linewidth = 1.5)
                                ax.fill_between(sub_data.index, sub_data['25%'], sub_data['75%'], facecolor = primary_color, alpha = 0.3)

                        first_index = max(itv)

                        if ((last_index >= max(itv)) & (i == len(self._rm) - 1)):
                            sub_data = data[data.index >= max(itv)]

                            if (first_plot):
                                ax.plot(sub_data.index, sub_data['mean'], color = secondary_color, linestyle = 'dashed', linewidth = 1, label = 'Mean')
                                ax.plot(sub_data.index, sub_data['50%'], color = primary_color, linewidth = 1.5, label = 'Median')
                                ax.fill_between(sub_data.index, sub_data['25%'], sub_data['75%'], facecolor = primary_color, alpha = 0.3, label = 'IQR Range')
                                first_plot = False

                            else:
                                ax.plot(sub_data.index, sub_data['mean'], color = secondary_color, linestyle = 'dashed', linewidth = 1)
                                ax.plot(sub_data.index, sub_data['50%'], color = primary_color, linewidth = 1.5)
                                ax.fill_between(sub_data.index, sub_data['25%'], sub_data['75%'], facecolor = primary_color, alpha = 0.3)

        else:
            ax.plot(data.index, data['mean'], color = secondary_color, linestyle = 'dashed', linewidth = 1, label = 'Mean')
            ax.plot(data.index, data['50%'], color = primary_color, linewidth = 1.5, label = 'Median')
            ax.fill_between(data.index, data['25%'], data['75%'], facecolor = primary_color, alpha = 0.3, label = 'IQR Range')

        ax.legend()
        ax.grid()
        plt.show()

    @property
    def data(self) -> pd.DataFrame:
        '''
        The original library data as a `pandas.DataFrame`.
        '''
        return self._d

    @data.setter
    def data(self, new: pd.DataFrame):
        self._d = new 
        self._t = self._get_stat()

    @property
    def stat(self) -> pd.DataFrame:
        '''
        Give the min, 25%, 50%, 75%, and max of the SpectralLib on each wavelength.
        '''
        return self._t

    @property
    def schema(self) -> dict:
        '''
        The original schema file as a `dict`.
        '''
        return self._s

class SpectralLibCollection():

    def __init__(self, libs: dict[SpectralLib]) -> None:
        '''
        Create an instance of `SpectralLibCollection`. The arguments should be the instances of `SpectralLib`.
        '''
        self._c = dict()
        for k, v in libs.items():
            if isinstance(v, SpectralLib):
                self._c[k] = v

            else:
                raise ValueError('Only SpectralLib is accepted in the arguments')

        self._t = self._get_stat()
        self._m = '+1'
    
    def _get_stat(self) -> pd.DataFrame:
        '''
        Give the min, 25%, 50%, 75%, and max of the contents in the `SpectralLibCollection` on each wavelength.
        '''
        content = pd.DataFrame()
        for k, v in self._c.items():
            temp = v.stat.copy()
            temp.columns = pd.MultiIndex.from_product([[k], temp.columns])
            content = pd.concat([content, temp], axis = 1)

        return content

    def __getitem__(self, index) -> SpectralLib:
        return self._c[index]

    def __len__(self) -> int:
        return len(self._c.keys())

    def __repr__(self) -> str:
        items = [k for k, v in self._c.items()]

        if (len(items) > 6):
            items_str = f'[{items[0]}, {items[1]}, {items[2]}, ... , {items[-3]}, {items[-2]}, {items[-1]}]'
        else:
            items_str = f'[{", ".join(items)}]'
        
        return f'SpectralLibCollection of {len(items)} SpectralLib: {items_str}'

    def apply(self, action: dict) -> 'SpectralLibCollection':
        '''
        Apply function to all members in the `SpectralLibCollection`.
        '''
        content = dict()
        for k, v in self._c.items():
            content[k] = v.apply(action)
        
        return SpectralLibCollection(content)

    def differential(self) -> 'SpectralLibCollection':
        '''
        Calculate the differential of the original spectra.
        '''
        content = dict()
        for k, v in self._c.items():
            temp = v.differential()
            content[k] = temp

        obj = SpectralLibCollection(content)
        obj._m = '-1'

        return obj

    def low_pass_filter(self, gap: int) -> 'SpectralLibCollection':
        '''
        Apply a low-pass filter to y-axis over x-axis, accoording to the `gap` on x-axis. (Basically a moving average filter.)
        '''
        content = dict()
        for k, v in self._c.items():
            temp = v.low_pass_filter(gap)
            content[k] = temp

        return SpectralLibCollection(content)

    def stat_plots(self, colors: list[str] = ['tab:blue', 'tab:orange', 'tab:green']):
        '''
        Give the plot according to `SpectralLibCollection.stat`.
        '''
        data = self._t
        counter = 0
        fig, ax = plt.subplots()
        if (self._m == '+1'):
            ax.set(xlabel = 'Wavelength (nm)', ylabel = 'Reflectance', ylim = (0, 0.5), title = 'All Spectra Distibution')

        elif (self._m == '-1'):
            # ax.set(xlabel = 'Wavelength (nm)', ylabel = 'Slope of Reflectance', ylim = (-0.01, 0.01), title = 'All Spectra Slope Distibution')
            ax.set(xlabel = 'Wavelength (nm)', ylabel = 'Slope of Reflectance', ylim = (-0.001, 0.001), title = 'All Spectra Slope Distibution')

        for k, v in self._c.items():
            if isinstance(v._rm, list):
                first_index = data.index.min()
                last_index = data.index.max()
                first_plot = True
                for i, itv in enumerate(v._rm):
                    col = counter % len(colors)
                    if isinstance(itv, tuple):
                            if (first_index <= min(itv)):
                                sub_data = data[k][(data.index >= first_index)&(data.index <= min(itv))]

                                if (first_plot):
                                    ax.plot(sub_data.index, sub_data['50%'], color = colors[col], linewidth = 1.5, label = f'{k} Median')
                                    ax.fill_between(sub_data.index, sub_data['25%'], sub_data['75%'], facecolor = colors[col], alpha = 0.3)
                                    first_plot = False

                                else:
                                    ax.plot(sub_data.index, sub_data['50%'], color = colors[col], linewidth = 1.5)
                                    ax.fill_between(sub_data.index, sub_data['25%'], sub_data['75%'], facecolor = colors[col], alpha = 0.3)

                            first_index = max(itv)

                            if ((last_index >= max(itv)) & (i == len(v._rm) - 1)):
                                sub_data = data[k][data.index >= max(itv)]

                                if (first_plot):
                                    ax.plot(sub_data.index, sub_data['50%'], color = colors[col], linewidth = 1.5, label = 'Median')
                                    ax.fill_between(sub_data.index, sub_data['25%'], sub_data['75%'], facecolor = colors[col], alpha = 0.3)
                                    first_plot = False

                                else:
                                    ax.plot(sub_data.index, sub_data['50%'], color = colors[col], linewidth = 1.5)
                                    ax.fill_between(sub_data.index, sub_data['25%'], sub_data['75%'], facecolor = colors[col], alpha = 0.3)

            else:
                ax.plot(data.index, data[k]['50%'], color = colors[col], linewidth = 1.5, label = f'{k} Median')
                ax.fill_between(data.index, data[k]['25%'], data['75%'], facecolor = colors[col], alpha = 0.3)

            counter += 1

        ax.fill_between(data.index, pd.Series(0, index = data.index), pd.Series(0, index = data.index), facecolor = 'grey', alpha = 0.3, label = 'IQR Range')
        ax.legend()
        ax.grid()
        plt.show()

    @property
    def items(self) -> list:
        '''
        The content of the `SpectralLibCollection`.
        '''
        content = list()
        for k, v in self._c.items():
            content.append((k, v))

        return content
    
    @items.setter
    def items(self, new: list[tuple[str, SpectralLib]]) -> None:
        content = dict()
        for k, v in new:
            if isinstance(v, SpectralLib):
                content[k] = v

            else:
                raise ValueError('Only SpectralLib is accepted in the arguments')
            
        self._c = content
        self._t = self._get_stat()

    @property
    def mean(self) -> pd.DataFrame:
        '''
        Give a quick view on the mean value of all `SpectralLib`.
        '''
        content = pd.DataFrame()
        for k, v in self._c.items():
            temp = v.data.mean(axis = 1).rename(k)
            content = pd.concat([content, temp], axis = 1)
        
        return content

    @property
    def stat(self) -> pd.DataFrame:
        '''
        Give the min, 25%, 50%, 75%, and max of the contents in the `SpectralLibCollection` on each wavelength.
        '''
        return self._t