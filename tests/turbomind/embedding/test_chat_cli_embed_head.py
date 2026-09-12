# tests/turbomind/embedding/test_chat_cli_embed_head.py
"""`lmdeploy chat` must forward --embed-head / --embed-head-format to TurboMind."""
import pytest

from lmdeploy import TurbomindEngineConfig
from lmdeploy.cli.chat import build_pipe
from lmdeploy.cli.cli import CLI
from lmdeploy.cli.utils import convert_args


@pytest.fixture(scope='module')
def chat_parser():
    CLI.add_parser_chat()
    return CLI.parser


def test_chat_cli_embed_head_flags_reach_turbomind_config(chat_parser, monkeypatch):
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr('lmdeploy.cli.chat.pipeline', fake_pipeline)

    args = chat_parser.parse_args([
        'chat',
        'test-model',
        '--backend',
        'turbomind',
        '--embed-head',
        'on',
        '--embed-head-format',
        'int4',
    ])
    kwargs = convert_args(args)
    build_pipe(kwargs.pop('model_path'), kwargs.pop('backend'), **kwargs)

    engine_config = captured['backend_config']
    assert isinstance(engine_config, TurbomindEngineConfig)
    assert engine_config.embed_head == 'on'
    assert engine_config.embed_head_format == 'int4'


def test_chat_cli_embed_head_defaults(chat_parser, monkeypatch):
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr('lmdeploy.cli.chat.pipeline', fake_pipeline)

    args = chat_parser.parse_args(['chat', 'test-model', '--backend', 'turbomind'])
    kwargs = convert_args(args)
    build_pipe(kwargs.pop('model_path'), kwargs.pop('backend'), **kwargs)

    engine_config = captured['backend_config']
    assert engine_config.embed_head == 'auto'
    assert engine_config.embed_head_format == 'native'


@pytest.mark.parametrize('removed', ['bf16', 'fp16'])
def test_chat_cli_rejects_removed_formats(chat_parser, removed):
    with pytest.raises(SystemExit):
        chat_parser.parse_args(['chat', 'test-model', '--embed-head-format', removed])
