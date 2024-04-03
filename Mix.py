from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from SpectralLib import SpectralLib

class Mixer():

    def __init__(self, s1: float, s2: float, s3: float, error: float = 0) -> None:
        '''
        Create an instance of `Mixer`. All three spectra must be provided.
        '''
        self.s1  = s1
        self.s2  = s2
        self.s3  = s3
        self.err = error

        def construct_matrix_cell(row_index: float, column_index: float) -> float:
            '''
            Construct vector cell contents according to the provided row & column index.
            '''
            a1 = row_index    / 100                         # Fill Fraction 1
            a2 = column_index / 100                         # Fill Fraction 2
            a3 = 1 - a1 - a2                                # Fill Fraction 3
            mask = np.where(a3.copy() >= 0, 1, np.NaN)      # a1 + a2 > 1 is invalid
            return (a1 * s1 + a2 * s2 + a3 * s3) * mask

        self.matrix = np.fromfunction(lambda i, j: construct_matrix_cell(i, j), (101, 101), dtype = float)

    def __repr__(self) -> str:
        return f'Mixer (s1 = {round(self.s1, 4)}\t, s2 = {round(self.s2, 4)}\t, s3 = {round(self.s3, 4)}\t, error = {round(self.err, 4)}\t)'
    
    def color_ramp(size: tuple = (101, 101, 4), mode: str = 'coord', x: float = None, y: float = None) -> np.ndarray:
        '''
        Generate the raibow-ish color ramp.
        '''
        ramp = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[0]):
                if ((size[0] - 1 - i - j) >= 0):
                    ramp[i, j] = [j / (size[0] - 1), i / (size[0] - 1), (size[0] - 1 - i - j) / (size[0] - 1), 1]   # (100, 0): Red, (0, 100): Green, (0, 0): Blue
                    
                else:
                    ramp[i, j] = [0, 0, 0, 0]

        if (mode == 'whole'):
            return ramp

        elif ((mode == 'coord') & (x is not None) & (y is not None)):
            return ramp[int(np.floor(y)), int(np.floor(x))]

    def color_scheme(title: str, x_axis_name: str, y_axis_name: str, figure_size: float = 6, fill: float = 2.5) -> None:
        '''
        Plot the color scheme reference. 
        '''
        size = (101, 101, 4)
        fig, ax = plt.subplots()
        ax.set(xlabel = f'Spectrum 2 - {x_axis_name} (%)', ylabel = f'Spectrum 1 - {y_axis_name} (%)', title = title)
        fig.set_size_inches(figure_size, figure_size)
        ramp = Mixer.color_ramp(mode = 'whole').reshape(-1, 4)
        ax.pcolormesh(range(0, size[0]), range(0, size[1]), np.zeros(size), color = ramp, lw = fill)
        plt.show()

    def error_plot(c1, c2, c3, names: dict, c4: list = [0, 0, 0, 0]):
        '''
        Plot the error matrix for the ingredient classification.
        '''
        labs = pd.DataFrame(np.array([c1[-1], c2[-1], c3[-1], c4]).T).rename(names, axis = 0).rename(names, axis = 1)
        labs['C. Tot'] = labs.sum(axis = 1)
        labs = labs.T
        labs['R. Tot'] = labs.sum(axis = 1)
        shape = (labs.shape[0] + 1, labs.shape[1] + 1)
        
        n  = int(labs.iloc[-1, -1])
        r1 = list()
        r2 = list()
        for i in range(labs.shape[0]):
            for j in range(labs.shape[1]):
                if ((i == j) & (i != labs.shape[0] - 1)):
                    r1.append(int(labs.iloc[i, j]))
                
                if ((j == labs.shape[1] - 1) & (i != labs.shape[0] - 1)):
                    r2.append(int(labs.iloc[i, j]) * int(labs.iloc[j, i]))

        kappa = round((n * sum(r1) - sum(r2)) / (n ** 2 - sum(r2)), 4)
        x_coords = np.linspace(0, 1, shape[1] + 1)
        y_coords = np.linspace(0, 1, shape[0] + 2)

        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)
        ax.set(xlim = (-0.08, 1.02), ylim = (-0.02, 1.02))

        for coord in x_coords:
            ax.vlines(coord, 0, 1 - 1 / (shape[0] + 1), colors = 'w')

        for coord in y_coords:
            ax.hlines(coord, 0, 1, colors = 'w')

        for i in range(shape[1]):
            for j in range(shape[0]):
                x0 = np.array([i * (1 / shape[1]), (i + 1) * (1 / shape[1])])
                y0 = np.array([(j + 1) * (1 / (shape[0] + 1)), (j + 1) * (1 / (shape[0] + 1))])
                y1 = np.array([j * (1 / (shape[0] + 1)), j * (1 / (shape[0] + 1))])

                if ((i == 0) | (j == shape[0] - 1)):
                    ax.fill_between(x0, y0, y1, fc = 'lightsteelblue', zorder = 0)

                else:
                    ax.fill_between(x0, y0, y1, fc = 'aliceblue', zorder = 0)

        for i, (k, v) in enumerate(labs.iterrows()):
            for j, (l, w) in enumerate(v.items()):
                ax.text((i + 1.5) / shape[1], 1 - (j + 2.5) / (shape[0] + 1), w, ha = 'center', va = 'center', color = 'lightslategray', size = 15, zorder = 0.1)
                if (i == 0):
                    ax.text(0.5 / shape[1], 1 - (j + 2.5) / (shape[0] + 1), l, ha = 'center', va = 'center', color = 'darkslategrey', size = 12, zorder = 0.1)

                if (j == 0):
                    ax.text((i + 1.5) / shape[1], 1 - (j + 1.5) / (shape[0] + 1), k, ha = 'center', va = 'center', color = 'darkslategrey', size = 12, zorder = 0.1)

        ax.text(-0.04, shape[1] / 2 / (shape[1] + 1), 'Classification Results', rotation = 'vertical', ha = 'center', va = 'center', color = 'darkslategrey', size = 10)
        ax.text(0.5, (shape[1] + 0.25) / (shape[1] + 1), 'Reference Spectra Type', ha = 'center', va = 'center', color = 'darkslategrey', size = 10)
        ax.text(0.5, 0.94, f'Error Matrix (K = {kappa})', ha = 'center', va = 'baseline', color = 'darkslategrey', size = 18)
        ax.set_axis_off()
        plt.show()

    def get_random_spectra_mix(n: int, s1: SpectralLib, s2: SpectralLib, s3: SpectralLib, seed: int = None) -> SpectralLib:
        '''
        Generate random spectra to test from; three spectra endmemebers must be provided.
        '''
        rng = np.random.default_rng(seed)
        s1_index = rng.integers(0, len(s1.data.columns), size = n)
        s2_index = rng.integers(0, len(s2.data.columns), size = n)
        s3_index = rng.integers(0, len(s3.data.columns), size = n)
        s1_proportions   = rng.uniform(0, 1, size = n)
        s2_proportions   = rng.uniform(0, 1 - s1_proportions)
        s3_proportions   = 1 - s1_proportions - s2_proportions
        s1_spectra = s1.data.iloc[:, s1_index].multiply(s1_proportions, axis = 1)
        s2_spectra = s2.data.iloc[:, s2_index].multiply(s2_proportions, axis = 1)
        s3_spectra = s3.data.iloc[:, s3_index].multiply(s3_proportions, axis = 1)
        s1_spectra.columns = [f'Random {i + 1}' for i in range(n)]
        s2_spectra.columns = [f'Random {i + 1}' for i in range(n)]
        s3_spectra.columns = [f'Random {i + 1}' for i in range(n)]
        rand_data = s1_spectra.add(s2_spectra, fill_value = 0).add(s3_spectra, fill_value = 0)

        rand = s1.copy()
        rand.data = rand_data
        rand.s1_prop = s1_proportions * 100
        rand.s2_prop = s2_proportions * 100
        rand.s3_prop = s3_proportions * 100
        return rand

    def ingredient(self, reflectance: float, output: str = 'info_array', use_weight: bool = True, weight_factor: float = 1, weight_base: float = 1) -> np.ndarray:
        '''
        Retrieve possible ingredient proportions from a given reflectance.
        '''
        qualified = ((self.matrix < (reflectance + self.err)) & (self.matrix > (reflectance - self.err)))
        if (output == 'mask'):
            return qualified
        
        elif (output == 'array'):
            return np.where(qualified)
        
        else:
            array = np.where(qualified)
            a3 = 100 - array[0] - array[1]
            result = (self.s1 * array[0] + self.s2 * array[1] + self.s3 * a3) / 100
            residual = result - reflectance
            if use_weight:
                weight = (1 - np.around(np.abs(residual) / self.err, 6)) * weight_factor + weight_base
                info = np.reshape(np.append(array, values = [a3, result, residual, weight]), (6, len(a3))).T

            else:
                info = np.reshape(np.append(array, values = [a3, result, residual]), (5, len(a3))).T
                
            if (output == 'info_dataframe'):
                return pd.DataFrame(info).rename({0: 's1', 1: 's2', 2: 's3', 3: 'result', 4: 'residual', 5: 'weight'}, axis = 1)
            
            elif (output == 'info_array'):
                return info
        
class MixerCollection():

    def __init__(self, s1: pd.Series = None, s2: pd.Series = None, s3: pd.Series = None, error: pd.Series = None) -> None:
        '''
        Create an instance of `MixerCollection`. The arguments should be all somehow iterable and share same length.
        '''
        self.s1  = s1
        self.s2  = s2
        self.s3  = s3
        self.err = error
        self.mixers = None
        if ((s1 is None) | (s2 is None) | (s3 is None)):
            pass

        else:
            if ((len(s1) != len(s2)) | (len(s1) != len(s3)) | (len(s2) != len(s3))):
                raise ValueError('All spectra should have same length.')
                
            elif not(s1.index.equals(s2.index) | s1.index.equals(s3.index) | s2.index.equals(s3.index)):
                raise ValueError('All spectra should have same indexing order.')
                
            else:
                if (error is None):
                    e = pd.Series(np.zeros(len(s1)), index = s1.index)

                else:
                    if (len(error) != len(s1)):
                        raise ValueError('Error should have same length with other provided spectra.')
                    
                    elif not(error.index.equals(s1.index)):
                        raise ValueError('Error should have same indexing order with other provided spectra.')

                    else:
                        e = error

                content = list()
                for i in s1.index:
                    r1 = s1[i]
                    r2 = s2[i]
                    r3 = s3[i]
                    content.append(Mixer(r1, r2, r3, e[i]))

                self.mixers = content

    def __repr__(self) -> str:
        nl = ',\n\t\t'
        items = [ f'  {x}' for x in self.mixers]
        parts = items
        if (len(items) > 6):
            parts = [x for xs in [[self.mixers[0]], items[1:3], ['  ...'], items[-3: ]] for x in xs]

        items_str = f'[{nl.join(map(str, parts))}]'
        return f'MixerCollection ({items_str})\n\nLength: {len(items)}'
    
    def comparing_plots(self, subject: SpectralLib, spectra_name: str, index_range: tuple = None, **kwargs) -> None:
        '''
        Give the plot that compares custom index's predict value and true value.
        '''
        analysis_result = list()
        start = 0
        ends  = len(subject.data.columns)
        figure_size   = 6
        status = False
        use_weight    = True
        weight_factor = 1
        weight_base   = 1
        if (index_range is not None):
            start = index_range[0]
            ends  = index_range[1]

        for k, v in kwargs.items():
            match k:
                case 'figure_size':
                    figure_size = v
                case 'status':
                    status = v
                case 'use_weight':
                    use_weight = v
                case 'weight_factor':
                    weight_factor = v
                case 'weight_base':
                    weight_base = v
                case _:
                    pass
        
        for i in range(start, ends):
            data = self.ingredient(subject.data.iloc[:, i], 'all', use_weight, weight_factor, weight_base)
            analysis_result.append(data['result'])
            if status:
                print(f'Executing {i + 1}/{ends}')

        comparing = pd.DataFrame([subject.s2_prop, np.array(analysis_result).T[1]]).T
        comparing.columns = ['Observed', 'Predicted']
        colors = pd.DataFrame([np.array(analysis_result).T[1], np.array(analysis_result).T[0]]).T
        colors['color'] = colors.apply(lambda row: Mixer.color_ramp(x = row.iloc[0], y = row.iloc[1]), axis = 1)
        rmse = round(np.sqrt(np.mean((comparing['Predicted'] - comparing['Observed']) ** 2)), 2)
        reg_b, reg_m = np.polynomial.polynomial.polyfit(comparing['Observed'], comparing['Predicted'].fillna(0), 1)
        cc_score = ((comparing['Observed'] - comparing['Observed'].mean()) * (comparing['Predicted'] - comparing['Predicted'].mean()) / comparing['Observed'].std() / comparing['Predicted'].std()).sum()
        cocoeff  = round(cc_score / (len(comparing) - 1), 4)

        fig, ax = plt.subplots()
        fig.set_size_inches(figure_size, figure_size)
        ax.set(xlabel = f'Observed {spectra_name} Coverage(%)', ylabel = f'Predicted {spectra_name} Coverage(%)', xlim = (0, 100), ylim = (0, 100), title = f'Simulating Custom Index Responding to Spectrum Data')
        ax.scatter(comparing['Observed'], comparing['Predicted'], color = colors['color'], marker = 'o', s = 10)
        ax.axline(xy1 = (0, reg_b), slope = reg_m, color = 'silver', ls = 'dashed')
        ax.text(95, 20, f'{len(comparing)} Spectra\nRMSE = {rmse}\nPCC = {cocoeff}', ha = 'right', va = 'top')
        plt.show()

    def ingredient(self, reflectance: pd.Series, output: str = 'all', use_weight: bool = True, weight_factor: float = 1, weight_base: float = 1) -> pd.Series:
        '''
        Retrieve possible ingredient proportions from a given reflectance.
        '''
        container    = dict()
        frequency_tb = pd.Series()
        for i, item in enumerate(self.mixers):
            comb = item.ingredient(reflectance.iloc[i], use_weight = use_weight, weight_factor = weight_factor, weight_base = weight_base)
            occurrance = np.full(len(comb), 1)
            weights    = comb.T[5]
            if use_weight:
                comb = [(s1, s2) for s1, s2, s3, rs, rd, wt in comb.tolist()]
                counter = pd.Series(weights, index = comb)

            else:
                comb = [(s1, s2) for s1, s2, s3, rs, rd in comb.tolist()]
                counter = pd.Series(occurrance, index = comb)

            frequency_tb = frequency_tb.add(counter, fill_value = 0)

            # Instead of appending all value to a series and find the mode,
            #     simply adding all occurance is way much faster than using
            #     pd.Series.values_count() or pd.Series.mode().

        if (output == 'frequency'):
            return frequency_tb

        try:
            top20 = frequency_tb[frequency_tb >= frequency_tb.sort_values(ascending = False).iloc[19]]
            # top50 = frequency_tb[frequency_tb >= frequency_tb.sort_values(ascending = False).iloc[49]]
        except:
            container['result'] = np.array([np.NAN, np.NAN, np.NAN])
            return container
        
        candidates = pd.DataFrame(top20.index.tolist()).rename({0: 's1', 1: 's2'}, axis = 1)
        candidates['s3'] = 100 - candidates['s1'] - candidates['s2']
        candidates['occurance'] = top20.values
        if (output == 'info_dataframe'):
            return candidates

        Spec1 = (candidates['s1'] * candidates['occurance']).sum() / candidates['occurance'].sum()
        Spec2 = (candidates['s2'] * candidates['occurance']).sum() / candidates['occurance'].sum()
        Spec3 = (candidates['s3'] * candidates['occurance']).sum() / candidates['occurance'].sum()
        mixed = np.array([Spec1, Spec2, Spec3])
        if (output == 'info_array'):
            return mixed
        
        if (output == 'all'):
            container['frequency']  = frequency_tb
            container['candidates'] = candidates
            container['result']     = mixed
            return container
        
    def ingredient_plots(self, subject: SpectralLib, spectra_names : list, spectra_index: int, threshold: float = 70, index_range: tuple = None, **kwargs) -> list:
        '''
        Plot SMA results on the scale according to `Mixer.color_scheme()`.
        '''
        analysis_result = list()
        classfication   = list()
        start = 0
        ends  = len(subject.data.columns)
        figure_size   = 6
        status = False
        use_weight    = True
        weight_factor = 1
        weight_base   = 1
        if (index_range is not None):
            start = index_range[0]
            ends  = index_range[1]

        for k, v in kwargs.items():
            match k:
                case 'figure_size':
                    figure_size = v
                case 'status':
                    status = v
                case 'use_weight':
                    use_weight = v
                case 'weight_factor':
                    weight_factor = v
                case 'weight_base':
                    weight_base = v
                case _:
                    pass
        
        for i in range(start, ends):
            data = self.ingredient(subject.data.iloc[:, i], 'all', use_weight, weight_factor, weight_base)
            analysis_result.append(data['result'])
            if status:
                print(f'Executing {i + 1}/{ends}')

        x_coords = np.array(analysis_result).T[1]
        y_coords = np.array(analysis_result).T[0]
        coords = pd.DataFrame([x_coords, y_coords]).T.rename({0: 'x', 1: 'y'}, axis = 1)
        coords['color'] = coords.apply(lambda row: Mixer.color_ramp(x = row.iloc[0], y = row.iloc[1]), axis = 1)

        match spectra_index:
            case 1:
                accuracy = round(len(coords[coords['y'] >= threshold]) / len(coords) * 100)
            case 2:
                accuracy = round(len(coords[coords['x'] >= threshold]) / len(coords) * 100)
            case 3:
                accuracy = round(len(coords[(coords['x'] <= 100 - threshold) & (coords['y'] <= 100 - threshold)]) / len(coords) * 100)

        reg1 = len(coords[coords['y'] >= threshold])
        reg2 = len(coords[coords['x'] >= threshold])
        reg3 = len(coords[(coords['x'] <= 100 - threshold) & (coords['y'] <= 100 - threshold)])
        reg4 = len(coords) - reg1 - reg2 - reg3
        classfication.append([reg1, reg2, reg3, reg4])

        R1  = np.array([0, 100 - threshold])
        R1r = np.array([100 - threshold, 0])
        R2  = np.array([threshold, 100])
        R2r = np.array([100, threshold])

        fig, ax = plt.subplots()
        fig.set_size_inches(figure_size, figure_size)
        ax.set(xlabel = f'Spectrum 2 - {spectra_names[1]} (%)', ylabel = f'Spectrum 1 - {spectra_names[0]} (%)', xlim = (0, 100), ylim = (0, 100), title = f'Classification of {spectra_names[spectra_index - 1]} Spectral Library')
        ax.scatter(coords['x'], coords['y'], color = coords['color'], marker = 'o', s = 30)
        ax.fill_between(R1, R2r, threshold, alpha = 0.2, fc = 'grey', zorder = 0)
        ax.fill_between(R1, R1r, 0 , alpha = 0.2, fc = 'grey', zorder = 0)
        ax.fill_between(R2, R1r, 0 , alpha = 0.2, fc = 'grey', zorder = 0)
        ax.text(45 - threshold / 2, threshold / 2 + 45, spectra_names[0], ha = 'center', va = 'center', color = 'silver', weight = 'bold', size = 14, zorder = 0.1)
        ax.text(threshold / 2 + 45, 45 - threshold / 2, spectra_names[1], ha = 'center', va = 'center', color = 'silver', weight = 'bold', size = 14, zorder = 0.1)
        ax.text(45 - threshold / 2, 45 - threshold / 2, spectra_names[2], ha = 'center', va = 'center', color = 'silver', weight = 'bold', size = 14, zorder = 0.1)
        ax.text(95, 95, f'{len(coords)} Spectra\nThreshold = {threshold}%\nAccuracy = {accuracy}%', ha = 'right', va = 'top')
        plt.show()
        return classfication