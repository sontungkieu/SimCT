"""Exercise real CPU aggregation without importing the GPU worker or its paths."""
import ast
from pathlib import Path
import unittest

from h2_word_metrics import aggregate, compare_distributions, distribution, js


class WordReportTests(unittest.TestCase):
    def test_probability_contract_collision_and_full_accounting(self):
        source=ast.parse(Path(__file__).with_name('h2_word_pilot.py').read_text(encoding='utf-8'))
        report=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='report')
        fixture={'key':'p0-r0-t8','prompt_id':'p0','category':'digit',
                 'sampling_weight':1,'candidates':['1','2']}
        data={'word-candidates.json':{'positions':[fixture]},'assets.json':{'models':{}}}
        for arm,masses in (('S',[.2,.1]),('B',[.18,.12]),('Q',[.1,.2])):
            data['word-scores-'+arm+'.json']={'rows':[{'key':fixture['key'],'masses':masses,
                'byte_probs':[1/258]*258,'token_telemetry':{}}]}
        outputs={}
        namespace={'read':data.__getitem__,'OUT':Path('/isolated-test/run'),
            'ARMS':('S','B','Q'),'CFG':{'e3_positions':1,'bootstrap_replicates':10},
            'compare_distributions':compare_distributions,'aggregate':aggregate,
            'distribution':distribution,'js':js,
            'write':lambda path,value:outputs.update({path.name:value}),
            'marker':lambda *args:None}
        exec(compile(ast.Module(body=[report],type_ignores=[]),'report-under-test','exec'),namespace)
        namespace['report']()
        summary=outputs['word-summary.json']
        self.assertEqual(summary['valid_positions'],1)
        self.assertEqual(summary['failures'],[])
        self.assertEqual(summary['X_raw']['valid_position_count'],1)
        row=outputs['word-metrics.json']['rows'][0]
        self.assertEqual(row['probability_contract'],
            'hybrid exact ASCII atomic marginal and canonical-word boundary proxy')
        self.assertIsNotNone(row['X_byte_raw'])
        self.assertIsNotNone(row['X_candidate_conditional'])
        self.assertEqual(row['dropped_candidates'],0)


if __name__=='__main__': unittest.main()
