# -*- coding: utf-8 -*-
# !/usr/bin/python
import numpy
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import manifold

import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class Drawer(object):

    def __init__(self):
        self.eps = numpy.finfo(float).eps

    def draw(self, input1, input2, name1, name2, figname, path_save):
        self.drawScatter(input1, input2, name1, name2, figname, path_save)
        print("drawing finished")

    def drawScatter(
        self, input1, input2, name1, name2, figname, path_save):
        """
        input1 and input2 : list of tuples (weighted score, weight)
        weight is unnormalized
        similar to input of Bootstrapping
        """
        print("drawing scatter plot")
        x, y = [], []
        for nume_1_deno_1, nume_2_deno_2 in zip(input1, input2):
            x.append(nume_1_deno_1[0] / nume_1_deno_1[1] )
            y.append(nume_2_deno_2[0] / nume_2_deno_2[1] )
        x, y = map(numpy.array, [x, y])

        plt.scatter(x, y, c='b', alpha=0.5, marker='o')
        plt.xlabel(name1)
        plt.ylabel(name2)

        minmin = min(x.min(), y.min())
        maxmax = max(x.max(), x.max())
        ax = plt.gca()
        linex = numpy.linspace(minmin, maxmax, 100)
        ax.plot(linex, linex, c='r')
        plt.savefig(
            os.path.join(path_save, f"scatter_{figname}.pdf"), format='pdf')

    def drawLearningCurve(self, columns, figname, draw_gold, draw_band, path_save): 
        """
        draw learning curve 
        columns : dict key : list of values 
        key includes : seqs, NHP, structured NHP, gold 
        """
        print("drawing learning curve")
        x = numpy.arange(len(columns['seqs']))
        if draw_gold: 
            plt.plot(x, columns['gold'], marker='', ls='-', color='r',  label="oracle")
        plt.plot(x, columns['structured NHP'], marker='o', ls='--', color='b', label='structured NHP')
        plt.plot(x, columns['NHP'], marker='v', ls='--', color='purple', label='NHP') #color='g',  label="NHP")
        # we may use plt.fill_between to get bands 
        # reason not to : 
        # such bands may overlap and implies resutls are not significant 
        # which may not true --- one may be consistenly better than the other by only a little 
        # it is still significant
        # pair perm test may show p-value < 0.01 
        if draw_band:
            plt.fill_between(x, columns['structured NHP low'], columns['structured NHP high'], color="b", alpha=0.2)
            plt.fill_between(x, columns['NHP low'], columns['NHP high'], color="g", alpha=0.2)
        #plt.title("Learning Curve")
        """
        {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        """
        plt.xlabel("number of training sequences", fontsize='x-large')
        plt.xticks(x, columns['seqs']) 
        plt.ylabel("log-likelihood per event", fontsize='x-large')
        plt.legend(loc="best", prop={'size': 'x-large'})
        plt.tight_layout()
        plt.savefig(
            os.path.join(path_save, f"learningcurve_{figname}.pdf"), format='pdf')


    def drawChart(self, columns, figname, path_save): 
        
        assert False, "deprecated! try drawChart2!"

        print("drawing chart ...")
        n_groups = 2

        tag_title = {
            'time': 'MSE %', 
            'type': 'error rate %'
        }

        # Get current size
        fig_size = plt.rcParams["figure.figsize"]
        # Prints: [8.0, 6.0]
        print(f"Current size: {fig_size}")
        fig_size[1] = fig_size[0] * 0.6
        plt.rcParams["figure.figsize"] = fig_size

        for eval_tag in ['time', 'type']: 
            print(f"tag is {eval_tag}")
            #to get the limits
            max_high = numpy.max(
                (
                    columns[eval_tag]['NHP high'], 
                    columns[eval_tag]['structured NHP high']
                )
            )
            min_low = numpy.min(
                (
                    columns[eval_tag]['NHP low'], 
                    columns[eval_tag]['structured NHP low']
                )
            )
            diff_max_min = max_high - min_low
            y_min = min_low - 0.2 * diff_max_min
            y_min = 0.0 if y_min <= 0.0 else y_min
            y_max = max_high + 0.2 * diff_max_min

            bar_up_men = ()
            bar_down_men = ()

            means_men = [
                columns[eval_tag]['NHP'], 
                columns[eval_tag]['structured NHP']
            ]
            highs_men = [
                columns[eval_tag]['NHP high'], 
                columns[eval_tag]['structured NHP high']
            ]
            lows_men = [
                columns[eval_tag]['NHP low'], 
                columns[eval_tag]['structured NHP low']
            ]

            for i, (mean, high, low) in enumerate(zip(means_men, highs_men, lows_men)):
                bar_up_men += (high-mean,)
                bar_down_men += (mean-low,)

            bars_mens = [bar_down_men, bar_up_men]
            error_config = {
                'ecolor': '0.3',
                'lw': 3, 'capsize': 15, 'capthick': 3
            }

            fig, ax = plt.subplots()

            index = numpy.arange(n_groups)
            #bar_width = 0.6

            opacity = 0.5

            result = ax.bar(
                index, means_men,
                yerr=bars_mens,
                width=0.8, 
                #align='edge', 
                #bar_width,
                alpha=opacity,
                error_kw=error_config,
                color=['green', 'blue'],
                linewidth=3
                #capsize=10, capthick=5
            )

            ax.margins(0.05)
            size_font = 20
            plt.xlabel(tag_title[eval_tag], fontsize=size_font)
            #if 'rmse' in file_save:
            #    plt.ylabel('RMSE', fontsize=size_font)
            #else:
            #    plt.ylabel('ErrorRate %', fontsize=size_font)
            #plt.title('Performance on synthetic dataset')
            size_font = 20
            plt.xticks(
                index+0.0,
                ('NHP', 'structured NHP'),
                fontsize = size_font
            )
            plt.yticks(fontsize = size_font)
            axes = plt.gca()
            axes.set_ylim([y_min,y_max])
            plt.legend()

            plt.tight_layout()

            def autolabel(rects):
                """
                Attach a text label above each bar displaying its height
                """
                for rect in rects:
                    height = rect.get_height()
                    ax.text(
                        rect.get_x() + rect.get_width()/2., 1.005*height,
                        '%d' % round(height, 2),
                        ha='center', va='bottom'
                    )

            #autolabel(result)
            #plt.show()
            plt.savefig(
                os.path.join(
                    path_save, 
                    f"chart_{eval_tag}_{figname}.pdf"
                ), 
                format='pdf'
            )


    def drawChart2(self, columns, figname, path_save): 

        tag_title = {
            'time-rmse': 'RMSE', 
            'time-mse': 'MSE', 
            'type': 'error rate %', 
            'nll': 'negative log likelihood'
        }

        model_color = {
            'NHP': 'purple', 
            'KnowEvolve': 'red', 
            'DyRep': 'green', 
            'DyRep++': 'teal', 
            'NDTT-': 'cornflowerblue', 
            'NDTT': 'blue', 
            'NoNeural': 'orange', 
        }
        model_short = {
            'NHP': 'NHP', 
            'KnowEvolve': 'KE', 
            'DyRep': 'DyRep', 
            'DyRep++': 'DyRep++', 
            'NDTT-': 'NDTT-', 
            'NDTT': 'NDTT', 
            'NoNeural': 'NoNeural'
        }

        all_possible_sorted = [
            'NHP', 'KnowEvolve', 'DyRep', 'DyRep++', 
            'NDTT-', 'NoNeural', 'NDTT'
        ]

        models = list()
        unsorted = set(columns['nll'].keys())
        for x in all_possible_sorted: 
            if x in unsorted: 
                models.append(x)

        print(f"drawing chart for {len(models)} models : {models}")
        n_groups = len(models)

        # Get current size
        fig_size = plt.rcParams["figure.figsize"]
        # Prints: [8.0, 6.0]
        print(f"Current size: {fig_size}")
        fig_size[1] = fig_size[0] * 0.6
        plt.rcParams["figure.figsize"] = fig_size

        for eval_tag in ['nll', 'time-mse', 'type', 'time-rmse']: 
            print(f"tag is {eval_tag}")

            means_men = [
                columns[eval_tag][model_name] for model_name in models
            ]

            highs_men = [
                columns[f'{eval_tag}-high'][model_name] for model_name in models
            ]

            lows_men = [
                columns[f'{eval_tag}-low'][model_name] for model_name in models
            ]

            color_list = [
                model_color[model_name] for model_name in models
            ]

            labels = tuple(
                [
                    model_short[model_name] for model_name in models
                ]
            )
            
            # to get limits 
            max_high = numpy.max(highs_men)
            min_low = numpy.min(lows_men)
            diff_max_min = max_high - min_low
            y_min = min_low - 0.2 * diff_max_min
            y_min = 0.0 if y_min <= 0.0 else y_min
            y_max = max_high + 0.2 * diff_max_min

            bar_up_men = ()
            bar_down_men = ()

            for i, (mean, high, low) in enumerate(zip(means_men, highs_men, lows_men)):
                bar_up_men += (high-mean,)
                bar_down_men += (mean-low,)

            bars_mens = [bar_down_men, bar_up_men]
            error_config = {
                'ecolor': '0.3',
                'lw': 3, 'capsize': 15, 'capthick': 3
            }

            fig, ax = plt.subplots()

            index = numpy.arange(n_groups)
            #bar_width = 0.6

            opacity = 0.5

            result = ax.bar(
                index, means_men,
                yerr=bars_mens,
                width=0.8, 
                alpha=opacity,
                error_kw=error_config,
                color=color_list,
                linewidth=3
                #capsize=10, capthick=5
            )

            ax.margins(0.05)
            size_font = 20
            plt.xlabel(tag_title[eval_tag], fontsize=size_font)
            
            size_font = 20
            plt.xticks(
                index+0.0,
                labels,
                fontsize = size_font
            )
            plt.yticks(fontsize = size_font)
            axes = plt.gca()
            axes.set_ylim([y_min,y_max])
            plt.legend()

            plt.tight_layout()

            def autolabel(rects):
                """
                Attach a text label above each bar displaying its height
                """
                for rect in rects:
                    height = rect.get_height()
                    ax.text(
                        rect.get_x() + rect.get_width()/2., 1.005*height,
                        '%d' % round(height, 2),
                        ha='center', va='bottom'
                    )

            #autolabel(result)
            #plt.show()
            plt.savefig(
                os.path.join(
                    path_save, 
                    f"chart2_{eval_tag}_{figname}.pdf"
                ), 
                format='pdf'
            )


class ClusterDrawer(object):

    def __init__(self):
        self.eps = numpy.finfo(float).eps

    def draw(self, embs, figname, path_save, mode, norm, size_ratio, config):
        if mode == 'pca': 
            model = PCA(n_components=2)
            self.draw_single(embs, figname, path_save, model, norm, size_ratio, config)
        elif mode == 'tsne': 
            plex = [1, 2, 4, 8, 16, 32, 64, 128]
            for p in plex: 
                model = manifold.TSNE(
                    n_components=2, init='random', random_state=0, 
                    perplexity=p
                )
                self.draw_single(
                    embs, f'{figname}_p={p}', path_save, model, norm, size_ratio, config)
        else: 
            raise Exception(f"Unknown mode : {mode}")
    
    def draw_single(self, embs, figname, path_save, model, norm, size_ratio, config): 
        """
        handle config
        """
        specs = dict()
        for line in config: 
            name, color, ratio, size = line.split(',')
            specs[name] = (color, ratio, size)
        #print(specs.keys())

        names = []
        x = []
        for k, v in embs.items(): 
            names.append(k)
            x.append(v)
        #assert len(names) == len(specs), f"names len : {len(names)} and specs len : {len(specs)}?!"

        X = numpy.array(x)
        # Standardize the data to have a mean of ~0 and a variance of 1
        if norm: 
            X = StandardScaler().fit_transform(X)
        # Create a PCA instance: pca
        components = model.fit_transform(X)
        # Plot the explained variances
        #features = range(pca.n_components_)
        #plt.bar(features, pca.explained_variance_ratio_, color='black')
        #plt.xlabel('PCA features')
        #plt.ylabel('variance %')
        #plt.xticks(features)
        fig, ax = plt.subplots()
        no_config_cnt = 0
        for i, txt in enumerate(names): 
            #print(specs[txt])
            if txt in specs: 
                ax.scatter(
                    components[i, 0], components[i, 1], 
                    alpha = 0.1 + 0.9 * float(specs[txt][1]), 
                    color = specs[txt][0], 
                    s = int(50 * float(specs[txt][2]) * size_ratio )
                )
            else: 
                no_config_cnt += 1
        #ax.scatter(components[:, 0], components[:, 1], alpha=.1, color='black')
        #for i, txt in enumerate(names):
        #    ax.annotate(txt, (components[i, 0], components[i, 1]))
        #plt.xlabel('PCA 1')
        #plt.ylabel('PCA 2')
        plt.savefig(
            os.path.join(
                path_save, 
                f"cluster_{figname}.pdf"
            ), 
            format='pdf'
        )
        print(f"finished drawing clusters")
        print(f"{no_config_cnt} points have no config info")