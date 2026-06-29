import 'package:flutter/material.dart';

import '../models/connection_settings.dart';
import '../services/token_client.dart';

enum _LoadState { loading, ok, error }

/// "What I remember about you" (Feature L): a read-only view of everything the server has stored
/// for this account — the distilled profile (summary + durable facts) plus recent conversation
/// turns — with a button to forget it all. The read half of the consent story; the write gate is
/// the "Remember our conversations" toggle in Settings.
///
/// Stored data shows even when recording consent is currently off (revoking keeps existing data),
/// captioned so that's clear — which is exactly what makes the Forget button meaningful.
class MemoryScreen extends StatefulWidget {
  const MemoryScreen({super.key, required this.settings});

  final ConnectionSettings settings;

  @override
  State<MemoryScreen> createState() => _MemoryScreenState();
}

class _MemoryScreenState extends State<MemoryScreen> {
  MemorySnapshot? _memory;
  _LoadState _state = _LoadState.loading;
  String? _error;
  bool _forgetting = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    if (mounted) setState(() => _state = _LoadState.loading);
    final client = TokenClient(widget.settings);
    try {
      final memory = await client.fetchMemory();
      if (!mounted) return;
      setState(() {
        _memory = memory;
        _state = _LoadState.ok;
        _error = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _state = _LoadState.error;
        _error = '$e';
      });
    } finally {
      client.close();
    }
  }

  Future<void> _forget() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Forget everything?'),
        content: const Text(
          'This permanently erases everything the assistant has stored about you on the '
          'server — facts, summary, and past conversations. This cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: Theme.of(context).colorScheme.error,
            ),
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Forget'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    setState(() => _forgetting = true);
    final client = TokenClient(widget.settings);
    try {
      final deleted = await client.deleteMemory();
      _snack(deleted ? 'Forgotten — nothing is stored now.' : 'There was nothing to forget.');
      await _load();
    } catch (e) {
      _snack('Could not forget: $e');
    } finally {
      client.close();
      if (mounted) setState(() => _forgetting = false);
    }
  }

  void _snack(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    final memory = _memory;
    final hasData = _state == _LoadState.ok && memory != null && !memory.isEmpty;
    return Scaffold(
      appBar: AppBar(
        title: const Text('What I remember'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Reload',
            onPressed: _state == _LoadState.loading ? null : _load,
          ),
        ],
      ),
      floatingActionButton: hasData
          ? FloatingActionButton.extended(
              onPressed: _forgetting ? null : _forget,
              backgroundColor: Theme.of(context).colorScheme.errorContainer,
              foregroundColor: Theme.of(context).colorScheme.onErrorContainer,
              icon: _forgetting
                  ? const SizedBox(
                      height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.delete_outline),
              label: const Text('Forget everything'),
            )
          : null,
      body: RefreshIndicator(onRefresh: _load, child: _body()),
    );
  }

  Widget _body() {
    switch (_state) {
      case _LoadState.loading:
        return const Center(child: CircularProgressIndicator());
      case _LoadState.error:
        return ListView(
          children: [
            const SizedBox(height: 80),
            Icon(Icons.cloud_off_outlined,
                size: 56, color: Theme.of(context).colorScheme.outline),
            const SizedBox(height: 16),
            Center(
              child: Text(_error ?? 'Could not load your memory.', textAlign: TextAlign.center),
            ),
            const SizedBox(height: 16),
            Center(child: FilledButton(onPressed: _load, child: const Text('Retry'))),
          ],
        );
      case _LoadState.ok:
        final memory = _memory!;
        if (memory.isEmpty) return _EmptyState(granted: memory.granted);
        return ListView(
          padding: const EdgeInsets.only(bottom: 96),
          children: [
            if (!memory.granted) const _ConsentOffBanner(),
            if (memory.summary.trim().isNotEmpty) _SummaryCard(summary: memory.summary),
            if (memory.facts.isNotEmpty) ...[
              _SectionHeader('Facts (${memory.facts.length})'),
              for (final fact in memory.facts) _FactTile(fact: fact),
            ],
            if (memory.turns.isNotEmpty) ...[
              _SectionHeader('Recent conversation (${memory.turns.length})'),
              for (final turn in memory.turns) _TurnTile(turn: turn),
            ],
          ],
        );
    }
  }
}

/// A short, human date from an ISO timestamp (e.g. "2026-06-29T10:00:00+00:00" -> "Jun 29, 2026").
/// Falls back to the raw string if it can't be parsed, and to '' when null.
String _shortDate(String? iso) {
  if (iso == null || iso.isEmpty) return '';
  final dt = DateTime.tryParse(iso);
  if (dt == null) return iso;
  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', //
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];
  final local = dt.toLocal();
  return '${months[local.month - 1]} ${local.day}, ${local.year}';
}

class _EmptyState extends StatelessWidget {
  const _EmptyState({required this.granted});
  final bool granted;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return ListView(
      children: [
        const SizedBox(height: 80),
        Icon(Icons.psychology_outlined, size: 56, color: scheme.outline),
        const SizedBox(height: 16),
        Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: Text(
              granted
                  ? "Nothing stored yet. Have a conversation and I'll remember the important bits."
                  : 'Memory is off, and nothing is stored. Turn on “Remember our conversations” '
                      'in Settings to let me remember you between calls.',
              textAlign: TextAlign.center,
              style: TextStyle(color: scheme.onSurfaceVariant),
            ),
          ),
        ),
      ],
    );
  }
}

/// Shown above stored data when recording consent is currently off: the data is retained but no
/// new turns are being saved. Makes the Forget button's purpose obvious.
class _ConsentOffBanner extends StatelessWidget {
  const _ConsentOffBanner();

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      margin: const EdgeInsets.fromLTRB(12, 12, 12, 0),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Icon(Icons.history_toggle_off, size: 18, color: scheme.onSurfaceVariant),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              "Memory is off, so nothing new is being saved — but this is what's still stored. "
              'Forget it below, or turn memory back on in Settings.',
              style: TextStyle(fontSize: 13, color: scheme.onSurfaceVariant),
            ),
          ),
        ],
      ),
    );
  }
}

class _SummaryCard extends StatelessWidget {
  const _SummaryCard({required this.summary});
  final String summary;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      margin: const EdgeInsets.fromLTRB(12, 16, 12, 4),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(Icons.notes_outlined, size: 20, color: scheme.primary),
            const SizedBox(width: 12),
            Expanded(child: Text(summary, style: const TextStyle(height: 1.4))),
          ],
        ),
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader(this.text);
  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 6),
      child: Text(
        text.toUpperCase(),
        style: Theme.of(context).textTheme.labelMedium?.copyWith(
              color: Theme.of(context).colorScheme.primary,
              letterSpacing: 0.8,
            ),
      ),
    );
  }
}

class _FactTile extends StatelessWidget {
  const _FactTile({required this.fact});
  final MemoryFact fact;

  @override
  Widget build(BuildContext context) {
    final date = _shortDate(fact.ts);
    return ListTile(
      dense: true,
      leading: const Icon(Icons.check_circle_outline, size: 20),
      title: Text(fact.text),
      subtitle: date.isEmpty ? null : Text('Since $date'),
    );
  }
}

class _TurnTile extends StatelessWidget {
  const _TurnTile({required this.turn});
  final RememberedTurn turn;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final isUser = turn.isUser;
    return ListTile(
      leading: CircleAvatar(
        radius: 16,
        backgroundColor: isUser ? scheme.secondaryContainer : scheme.primaryContainer,
        child: Icon(
          isUser ? Icons.person_outline : Icons.assistant_outlined,
          size: 18,
          color: isUser ? scheme.onSecondaryContainer : scheme.onPrimaryContainer,
        ),
      ),
      title: Text(turn.content),
      subtitle: Text(isUser ? 'You' : 'Assistant'),
    );
  }
}
