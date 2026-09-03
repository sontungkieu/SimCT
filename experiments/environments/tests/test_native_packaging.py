import ast
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_source', ROOT/'prepare_source.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


class NativePackagingTests(unittest.TestCase):
    def test_python_syntax(self):
        for path in ROOT.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))

    def test_launcher_regressions(self):
        script = (ROOT/'xtoken_smoke.py').read_text()
        self.assertIn("'--project',str(HERE/'xtoken')", script)
        self.assertIn('cwd=source', script)
        for owner in ('policy', 'teachers.0'):
            for field in ('force_hf=true', 'attn_implementation=sdpa'):
                self.assertIn(f'+{owner}.dtensor_cfg.automodel_kwargs.{field}', script)
        self.assertIn("validated.get('config_digest') == config_digest", script)

    def test_lock_identity(self):
        expected = {'xtoken': ('2.11.0+cu128', '5.12.1'), 'simct': ('2.9.1+cu128', '4.57.1')}
        for name, (torch, transformers) in expected.items():
            lock = tomllib.loads((ROOT/name/'uv.lock').read_text())
            packages = {p['name']: p for p in lock['package']}
            self.assertEqual(packages['torch']['version'], torch)
            self.assertEqual(packages['transformers']['version'], transformers)
            self.assertFalse(any(p.endswith('-cu13') for p in packages))

    def test_archive_pinned(self):
        lock = tomllib.loads((ROOT/'xtoken/uv.lock').read_text())
        package = next(p for p in lock['package'] if p['name'] == 'nemo-rl')
        self.assertIn('13a10647ebbf0f940d2b06ea41800b3f2fb46099', package['source']['url'])
        self.assertEqual(package['sdist']['hash'], 'sha256:4d8811064e04ccc90740eae8f2dd240ea501b0964f1eb559b9236a4491813fd0')

    def test_flash_attention_official_hash(self):
        lock = tomllib.loads((ROOT/'simct/uv.lock').read_text())
        package = next(p for p in lock['package'] if p['name'] == 'flash-attn')
        self.assertEqual(package['wheels'][0]['hash'], 'sha256:4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810')

    def test_source_verify_rejects_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            archive = temp/'source.tar.gz'
            payload = b'upstream source\n'
            with tarfile.open(archive, 'w:gz') as tar:
                info = tarfile.TarInfo('source/module.py')
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            tree = temp/'tree'
            tree.mkdir()
            (tree/'module.py').write_bytes(payload)
            self.assertEqual(source.verify_tree(archive, tree)['files_verified'], 1)
            (tree/'module.py').write_text('changed')
            with self.assertRaises(ValueError):
                source.verify_tree(archive, tree)

    def test_projects_independent(self):
        simct = tomllib.loads((ROOT/'simct/pyproject.toml').read_text())
        xtoken = tomllib.loads((ROOT/'xtoken/pyproject.toml').read_text())
        self.assertEqual(simct['tool']['uv']['sources']['kdflow']['path'], '../../..')
        self.assertNotIn('kdflow', xtoken['tool']['uv']['sources'])
        self.assertFalse(xtoken['tool']['uv']['package'])
        self.assertEqual(len(xtoken['tool']['uv']['exclude-dependencies']), 5)


if __name__ == '__main__':
    unittest.main()
