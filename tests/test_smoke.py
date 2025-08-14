def test_imports():
    import sirgraf
    from sirgraf.core import ProcessResult
    assert hasattr(sirgraf, "process_directory")
    assert ProcessResult is not None